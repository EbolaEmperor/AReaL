import argparse
import random
import sys

from datasets import load_dataset

from areal.api import AllocationMode
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.dataset import get_custom_dataset
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("GSM8KEval")


def _parse_runtime_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        default=None,
        help=(
            "Override model path for evaluation (also updates tokenizer/inference "
            "backend model path fields)."
        ),
    )
    parser.add_argument(
        "--n-shot",
        "--n_shot",
        dest="n_shot",
        type=int,
        default=0,
        help="Number of GSM8K in-context examples to prepend to each eval sample.",
    )
    parser.add_argument(
        "--fewshot-split",
        "--fewshot_split",
        dest="fewshot_split",
        type=str,
        default="train",
        help="Dataset split used to source few-shot demonstrations (default: train).",
    )
    parser.add_argument(
        "--fewshot-seed",
        "--fewshot_seed",
        dest="fewshot_seed",
        type=int,
        default=1,
        help="Random seed for few-shot index sampling when indices are not specified.",
    )
    parser.add_argument(
        "--fewshot-indices",
        "--fewshot_indices",
        dest="fewshot_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated source indices for few-shot examples. "
            "If provided, uses these indices in order."
        ),
    )
    return parser.parse_known_args(argv)


def _override_model_path(config: GRPOConfig, model_path: str | None) -> None:
    if not model_path:
        return

    config.actor.path = model_path
    config.tokenizer_path = model_path
    config.rollout.tokenizer_path = model_path
    config.sglang.model_path = model_path
    config.vllm.model = model_path

    if config.ref is not None:
        config.ref.path = model_path
    if config.critic is not None:
        config.critic.path = model_path


def _parse_indices(indices: str | None) -> list[int] | None:
    if indices is None:
        return None
    return [int(x) for x in indices.split(",") if x.strip()]


def _load_fewshot_examples(
    dataset_path: str,
    n_shot: int,
    source_split: str,
    seed: int,
    source_indices: list[int] | None,
) -> tuple[list[tuple[str, str]], list[int]]:
    if n_shot <= 0:
        return [], []

    source_dataset = load_dataset(path=dataset_path, name="main", split=source_split)
    total = len(source_dataset)
    if n_shot > total:
        raise ValueError(
            f"Requested n_shot={n_shot}, but source split '{source_split}' has only "
            f"{total} samples."
        )

    if source_indices is not None:
        if len(source_indices) < n_shot:
            raise ValueError(
                "Provided fewer few-shot indices than n_shot: "
                f"{len(source_indices)} < {n_shot}"
            )
        selected_indices = source_indices[:n_shot]
        for idx in selected_indices:
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"Few-shot index {idx} out of range for split '{source_split}' "
                    f"with size {total}."
                )
    else:
        rng = random.Random(seed)
        selected_indices = rng.sample(range(total), n_shot)

    fewshot_examples: list[tuple[str, str]] = []
    for idx in selected_indices:
        sample = source_dataset[idx]
        user_prompt = (
            sample["question"] + "\nPlease put your final answer within \\boxed{}."
        )
        fewshot_examples.append((user_prompt, sample["answer"]))

    return fewshot_examples, selected_indices


def _prepend_fewshot_messages(
    sample: dict,
    fewshot_examples: list[tuple[str, str]],
) -> dict:
    messages: list[dict[str, str]] = []
    for shot_question, shot_answer in fewshot_examples:
        messages.append({"role": "user", "content": shot_question})
        messages.append({"role": "assistant", "content": shot_answer})
    messages.extend(sample["messages"])
    return {"messages": messages}


def _extract_accuracy(eval_stats: dict[str, float]) -> float | None:
    """Extract GSM8K exact-match accuracy from exported stats.

    RLVR workflow records this metric as reward under either
    `eval-rollout/reward` or `reward` depending on stat scope/export path.
    """
    preferred_keys = ["eval-rollout/reward", "reward"]
    for key in preferred_keys:
        value = eval_stats.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    # Fallback: find a key that ends with `/reward`.
    for key, value in eval_stats.items():
        if key.endswith("/reward") and isinstance(value, (int, float)):
            return float(value)
    return None


def main(args):
    runtime_args, remaining_args = _parse_runtime_args(args)
    config, _ = load_expr_config(remaining_args, GRPOConfig)
    _override_model_path(config, runtime_args.model_path)
    if runtime_args.n_shot < 0:
        raise ValueError(f"--n-shot must be >= 0, got {runtime_args.n_shot}")

    if runtime_args.model_path:
        logger.info(
            f"Model path overridden via --model-path: {runtime_args.model_path}"
        )

    logging.setup_file_logging(config.cluster.fileroot, filename="eval.log")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="eval")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    source_indices = _parse_indices(runtime_args.fewshot_indices)
    fewshot_examples, fewshot_selected_indices = _load_fewshot_examples(
        dataset_path=config.valid_dataset.path,
        n_shot=runtime_args.n_shot,
        source_split=runtime_args.fewshot_split,
        seed=runtime_args.fewshot_seed,
        source_indices=source_indices,
    )
    if fewshot_examples:
        logger.info(
            f"Few-shot enabled: n_shot={len(fewshot_examples)}, "
            f"source_split={runtime_args.fewshot_split}, "
            f"indices={fewshot_selected_indices}"
        )

    # Initialize scheduler
    cfg = config.scheduler
    if cfg.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif cfg.type == "ray":
        scheduler = RayScheduler(exp_config=config)
    elif cfg.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")

    # Load evaluation dataset
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    if fewshot_examples:
        valid_dataset = valid_dataset.map(
            lambda sample: _prepend_fewshot_messages(sample, fewshot_examples),
            desc=f"Inject {len(fewshot_examples)}-shot exemplars",
            load_from_cache_file=False,
        )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )

    # Initialize RolloutController
    config.rollout.max_head_offpolicyness = int(1e12)

    if allocation_mode.gen_backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=allocation_mode.gen.tp_size,
            base_gpu_id=0,
        )
    elif allocation_mode.gen_backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=allocation_mode.gen.tp_size,
            pp_size=allocation_mode.gen.pp_size,
        )
    else:
        raise ValueError(f"Invalid backend: {allocation_mode.gen_backend}")

    eval_rollout = engine_cls.as_controller(config.rollout, scheduler)

    try:
        eval_rollout.initialize(
            role="eval-rollout",
            alloc_mode=allocation_mode,
            server_args=server_args,
        )

        # Create evaluation workflow
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
        )

        # Submit all evaluation tasks
        cnt = 0
        for data in valid_dataloader:
            for item in data:
                eval_rollout.submit(
                    item,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.gconfig.n_samples,
                )
                cnt += 1

        eval_rollout.wait(cnt, timeout=None)
        eval_stats = eval_rollout.export_stats()

        # Print and log results
        logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")

        accuracy = _extract_accuracy(eval_stats)
        if accuracy is None:
            logger.warning(
                "Could not infer GSM8K accuracy from eval stats keys: "
                f"{list(eval_stats.keys())}"
            )
        else:
            logger.info(
                "GSM8K exact-match accuracy (math_verify): "
                f"{accuracy:.4%} ({accuracy:.6f})"
            )
    finally:
        eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
