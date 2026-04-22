# 细粒度中文仇恨言论识别 (CCL25-Eval Task 10) 实验记录

## 任务

基于 STATE-ToxiCN 数据集，输入社交媒体文本，输出仇恨四元组：
`Target | Argument | Targeted Group | Hateful [END]`

评测指标：`(Hard_F1 + Soft_F1) / 2`
- Hard: 四字段完全一致
- Soft: Group + Hateful 完全一致，Target/Argument 的 `difflib.SequenceMatcher.ratio() > 0.5`

## 数据

- `train.json`: 4000 条标注样本，shuffle(seed=42) 后 90/10 划分 train/dev
- Train: 3600 条，Dev: 400 条
- 测试集 `test1.json` / `test2.json` 无标签（比赛提交用）

## 模型

- Qwen3-1.7B: `/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3-1___7B`
- Qwen3-4B: `/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3-4B`
- Qwen3-8B: `/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3-8B`

## 最终结果汇总

| 排名 | 模型 | 方法 | Hard F1 | Soft F1 | **Avg F1** |
|------|------|------|---------|---------|-----------|
| 🥇 | Qwen3-8B | SFT 3ep | 0.1521 | 0.3421 | **0.2471** |
| 🥈 | Qwen3-4B | SFT 3ep | 0.1455 | 0.3326 | **0.2391** |
| 🥉 | Qwen3-1.7B | SFT 3ep | 0.1145 | 0.2545 | **0.1845** |
| 4 | Qwen3-8B | GRPO ~700步 from base (3-shot) | 0.0824 | 0.2082 | **0.1453** |
| 5 | Qwen3-8B | base 3-shot | 0.0409 | 0.2065 | **0.1237** |
| 6 | Qwen3-8B | base 0-shot | — | — | ~0.03 (est) |
| 7 | Qwen3-4B | base 3-shot | 0.0213 | 0.1215 | **0.0714** |
| 8 | Qwen3-1.7B | base 3-shot | 0.0110 | 0.0644 | **0.0377** |
| 9 | Qwen3-4B | base 0-shot | 0.0096 | 0.0410 | **0.0253** |
| 10 | Qwen3-1.7B | base 0-shot | 0.0000 | 0.0078 | **0.0039** |

## 关键发现

1. **SFT >> GRPO**：SFT 3 epoch 在所有模型规模上都大幅优于 GRPO（8B: 0.25 vs 0.15）
2. **模型规模有帮助但边际递减**：1.7B→4B SFT +30%，4B→8B SFT +3.3%
3. **3-shot prompt 对 base 模型帮助巨大**：parse 失败率从 86% 降到 0-9%
4. **GRPO 从 base 训练需要很多步**：~700 步才从 0.12 升到 0.15，且后期趋于平台
5. **SFT 后接 GRPO 无收益**（1.7B 实验确认）：reward 过严 + 模型已接近 SFT 能达到的上限

## Checkpoints

- 🏆 **最佳**: 8B SFT v1 (已删除，需重训；或见下方保留的 GRPO ckpt)
- GRPO 8B base trial0: `/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-grpo-8b-base/trial0/default/` (step 349/399/449/499)
- GRPO 8B base trial1 (resumed): `/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-grpo-8b-base/trial1/default/` (step 49/99/149/199)

## 代码文件

| 文件 | 用途 |
|------|------|
| `areal/dataset/hate_recognition.py` | 数据集 loader (SFT + RL, 支持 3-shot) |
| `areal/reward/hate_recognition.py` | 官方 reward (difflib soft match) |
| `examples/hate_recogonization/eval.py` | 评测脚本 (--fewshot, --ckpt) |
| `examples/hate_recogonization/hate_recognition_rl.py` | GRPO 训练入口 |
| `examples/hate_recogonization/hate_recognition_sft.py` | SFT 训练入口 |
| `examples/hate_recogonization/hate_recognition_grpo_8b_base.yaml` | 8B GRPO 配置 |
| `examples/hate_recogonization/hate_recognition_sft_8b.yaml` | 8B SFT 配置 |

## Reward 设计迭代

1. **v1**: `format_bonus(0.1) + avg(hard, soft)` → 模型 hack format bonus，卡死 0.10
2. **v2**: 移除 format_bonus + 加 group_match → 模型抄袭 prompt 示例
3. **v3**: 移除示例 + 加 "必须输出 quad" 规则 → 8B base 改善
4. **v4 (最终)**: `avg(hard_F1, soft_F1)` + difflib.SequenceMatcher + 鲁棒 parser

## 环境踩坑

- liu001 网络封闭（modelscope/HF 不通），需本地下载后 rsync
- sgl_kernel ABI 不兼容 torch 2.10 → 用 vLLM 0.17.0
- flashinfer 版本不匹配 → `FLASHINFER_DISABLE_VERSION_CHECK=1`
- 8B GRPO 显存紧张 → `fsdp.offload_params=true` + `per_layer_optim_step=true`
- 长时间训练中途 OOM kill → `saver.freq_steps: 50` 防止丢进度

## 后续方向

- SFT + GRPO 联合训练（先 SFT 教格式，再 GRPO 教精度）需更宽松的 reward 或分阶段 reward
- 更多 SFT epoch（当前 3ep loss 仍有下降空间）
- 数据增强（用大模型生成更多标注）
- 换 CoT 范式（先思考再输出，利用 Qwen3 thinking mode）
