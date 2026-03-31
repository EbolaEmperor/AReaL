import hashlib
import re
import threading

from areal.utils import logging

logger = logging.getLogger("MatlabAIDetectReward")

_VERDICT_RE = re.compile(r"<verdict>\s*(ai|human)\s*</verdict>", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:matlab)?\n(.*?)\n```", re.DOTALL)
_BACKTICK_RE = re.compile(r"`([^`\n]{3,80})`")
_BOLD_RE = re.compile(r"\*\*([^*\n]{2,30})\*\*")

# Per-sample prediction history: key → list[bool]
# Keyed by MD5 of the prompt content, persists for the lifetime of the process.
_SAMPLE_HISTORY: dict[str, list[bool]] = {}
_HISTORY_LOCK = threading.Lock()

_MIN_HISTORY = 10       # minimum records before adaptive scaling kicks in
_HARD_THRESHOLD = 0.1   # correct_rate ≤ 10%  → hard sample
_EASY_THRESHOLD = 0.9   # correct_rate ≥ 90%  → easy sample


def _sample_key(prompt) -> str:
    return hashlib.md5(str(prompt).encode()).hexdigest()


def _extract_code(prompt) -> str:
    """Extract MATLAB code from the conversation prompt (list of messages or string)."""
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = msg.get("content", "")
                m = _CODE_BLOCK_RE.search(text)
                if m:
                    return m.group(1)
    elif isinstance(prompt, str):
        m = _CODE_BLOCK_RE.search(prompt)
        if m:
            return m.group(1)
    return ""


def _faithfulness_penalty(completion: str, code: str) -> float:
    """Penalize reasoning that cites features that don't exist in the code.

    Checks:
    1. Comment hallucination: model discusses comments (注释) but code has none (no %).
    2. Fabricated backtick quotes: model quotes specific code snippets that don't
       appear verbatim in the source.

    Returns a penalty in [-0.5, 0.0].
    """
    if not code:
        return 0.0

    penalty = 0.0

    # Rule 1: Comment hallucination
    # Extract the reasoning part (before the verdict tags)
    reasoning = re.split(r"<verdict>", completion, maxsplit=1)[0]
    has_comments_in_code = "%" in code
    cites_comments = bool(re.search(r"注释|comment", reasoning, re.IGNORECASE))
    if cites_comments and not has_comments_in_code:
        penalty -= 0.3
        logger.debug("Faithfulness penalty: cited comments but code has none")

    # Rule 2: Fabricated backtick-quoted code snippets
    quoted_snippets = _BACKTICK_RE.findall(reasoning)
    for snippet in quoted_snippets:
        # Only check snippets that look like code (contain MATLAB-like chars)
        if re.search(r"[a-zA-Z_]\w*|[=\+\-\*\/\(\)]", snippet) and snippet not in code:
            penalty -= 0.1
            logger.debug("Faithfulness penalty: fabricated snippet %r", snippet)

    return max(-0.5, penalty)


def _parse_output(text: str) -> tuple[str | None, float | None]:
    """Extract verdict and confidence from model output.

    Requires exactly one <verdict> tag and exactly one <confidence> tag.
    Multiple occurrences of either tag are treated as a format error and
    return (None, None).

    Returns (verdict, confidence) where verdict is 'ai'/'human' or None,
    and confidence is a float in [0, 1] or None.
    """
    verdict_matches = _VERDICT_RE.findall(text)
    if len(verdict_matches) != 1:
        return None, None
    verdict = verdict_matches[0].lower()

    confidence_matches = _CONFIDENCE_RE.findall(text)
    if len(confidence_matches) != 1:
        return None, None
    try:
        confidence = float(confidence_matches[0])
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        return None, None

    return verdict, confidence


def matlab_ai_detect_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Reward function with format reward, confidence-adjusted accuracy reward,
    and adaptive per-sample difficulty scaling.

    Base reward:
      - No valid format:       0.0
      - Valid format:         +0.1  (format reward)
      - Correct verdict:      +1.0 - (confidence - 1)^2  (range [0.1, 1.0], max at confidence=1.0)
      - Wrong verdict:        -1.0 - confidence^2         (range [-2.0, -1.0], min at confidence=1.0)
      Base total range: [-1.9, 1.1]
      Optimal confidence = true per-sample accuracy (Brier score calibration)

    Length penalty (applied after adaptive scaling):
      - If completion tokens > 800: deduct 0.05 per 150 tokens over the limit,
        capped at -0.5.

    Adaptive scaling (requires ≥ 10 recorded predictions for this sample):
      - Hard sample (correct_rate ≤ 10%):  reward × 2
      - Easy sample (correct_rate ≥ 90%):
          correct prediction → reward × 0.5
          wrong prediction   → reward × 2
    """
    try:
        verdict, confidence = _parse_output(str(completions))
        key = _sample_key(prompt)

        if verdict is None or confidence is None:
            # Format failure counts as a wrong prediction in history
            with _HISTORY_LOCK:
                _SAMPLE_HISTORY.setdefault(key, []).append(False)
            return 0.0

        format_reward = 0.1
        ground_truth = str(answer).strip().lower()
        is_correct = verdict == ground_truth

        # Brier score calibration: optimal solution is confidence = p_sample
        # penalizes (confidence - label)^2, where label=1.0 if correct, 0.0 if wrong
        label = 1.0 if is_correct else 0.0
        brier_penalty = (confidence - label) ** 2  # in [0, 1]
        content_reward = (1.0 if is_correct else -1.0) - brier_penalty

        reward = format_reward + content_reward

        # Faithfulness penalty: penalize reasoning that cites non-existent features
        code = _extract_code(prompt)
        reward += _faithfulness_penalty(str(completions), code)

        # Adaptive difficulty scaling
        with _HISTORY_LOCK:
            history = _SAMPLE_HISTORY.setdefault(key, [])
            n = len(history)
            if n >= _MIN_HISTORY:
                correct_rate = sum(history) / n
                if correct_rate <= _HARD_THRESHOLD:
                    reward *= 2.0
                elif correct_rate >= _EASY_THRESHOLD:
                    if is_correct:
                        reward *= 0.5
                    else:
                        reward *= 2.0
            history.append(is_correct)

        # Length penalty: -0.05 per 150 tokens over 800, capped at -0.5
        if completion_ids is not None:
            n_tokens = len(completion_ids)
            if n_tokens > 800:
                over = n_tokens - 800
                penalty = min(0.5, (over // 150 + 1) * 0.05)
                reward -= penalty

        return reward
    except Exception:
        logger.warning("Exception in matlab_ai_detect_reward_fn", exc_info=True)
        return 0.0
