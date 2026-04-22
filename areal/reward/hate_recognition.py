"""Reward function for fine-grained Chinese hate speech quadruple extraction.

Implements the official CCL25-Eval Task 10 metric: average of hard-match F1 and
soft-match F1 over predicted quadruples vs. gold.

Output format expected from the model:
    Target | Argument | Group | Hateful [END]
Multiple quadruples are separated by ` [SEP] ` and the full string ends with ` [END]`.

Match definitions (official):
  - hard match: every one of the 4 fields equals the gold field exactly
    (Group is normalised as a set of labels so order doesn't matter).
  - soft match: Targeted Group and Hateful are exactly equal, AND the string
    similarity of Target and Argument both EXCEED 0.5. Similarity uses
    Python's ``difflib.SequenceMatcher.ratio()`` — 2M / (|a| + |b|), where
    M is the total length of matching blocks.

Per-sample reward = (F1_hard + F1_soft) / 2 in [0, 1].
"""

from difflib import SequenceMatcher

from areal.utils import logging

logger = logging.getLogger("HateRecognitionReward")

_SOFT_THRESHOLD = 0.5  # similarity strictly greater than this is required
_VALID_GROUPS = {"Racism", "Region", "Sexism", "LGBTQ", "others", "non-hate"}
_VALID_HATEFUL = {"hate", "non-hate"}


def _extract_answer_region(text: str) -> str:
    """Trim the text to the last occurrence of `[END]` (inclusive)."""
    idx = text.rfind("[END]")
    if idx == -1:
        return text.strip()
    return text[: idx + len("[END]")].strip()


def _parse_quads(text: str) -> list[tuple[str, str, frozenset, str]] | None:
    """Parse an output string into a list of 4-tuples.

    Robust extraction:
      - Lone ``[END]`` (model refused / claimed no hate) -> ``[]`` (valid empty).
      - Missing ``[END]`` entirely -> fall back to parsing the whole string.
      - Individual malformed quads are skipped rather than failing the whole output.
    Returns ``None`` only when the gold-output side is empty (which should never
    happen for a labelled sample).
    """
    text = text.strip()
    if "[END]" in text:
        body = text[: text.rfind("[END]")].strip()
    else:
        body = text
    if not body:
        return []  # model explicitly produced nothing before [END]

    raw_quads = [q.strip() for q in body.split("[SEP]")]
    result = []
    for q in raw_quads:
        parts = [p.strip() for p in q.split("|")]
        if len(parts) != 4:
            continue
        target, argument, group, hateful = parts
        if not target or not argument or not group or not hateful:
            continue
        groups = frozenset(g.strip() for g in group.split(",") if g.strip())
        if not groups or not groups.issubset(_VALID_GROUPS):
            continue
        if hateful not in _VALID_HATEFUL:
            continue
        result.append((target, argument, groups, hateful))
    return result


def _similarity(a: str, b: str) -> float:
    """Official similarity: difflib.SequenceMatcher.ratio() = 2M / (|a| + |b|)."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _hard_match(p, g) -> bool:
    return p[0] == g[0] and p[1] == g[1] and p[2] == g[2] and p[3] == g[3]


def _soft_match(p, g) -> bool:
    if p[2] != g[2] or p[3] != g[3]:
        return False
    return (
        _similarity(p[0], g[0]) > _SOFT_THRESHOLD
        and _similarity(p[1], g[1]) > _SOFT_THRESHOLD
    )


def _greedy_f1(pred, gold, matcher) -> float:
    if not pred or not gold:
        return 0.0
    used = [False] * len(gold)
    tp = 0
    for p in pred:
        for j, g in enumerate(gold):
            if not used[j] and matcher(p, g):
                used[j] = True
                tp += 1
                break
    denom = len(pred) + len(gold)
    return 2 * tp / denom if denom else 0.0


def hate_recognition_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Reward: avg(hard_F1, soft_F1). No format bonus, no group partial credit."""
    try:
        completion_text = _extract_answer_region(str(completions))
        pred_quads = _parse_quads(completion_text)
        gold_quads = _parse_quads(_extract_answer_region(str(answer)))

        if gold_quads is None:
            logger.warning("Failed to parse gold answer: %r", answer)
            return 0.0

        if pred_quads is None:
            return 0.0

        hard_f1 = _greedy_f1(pred_quads, gold_quads, _hard_match)
        soft_f1 = _greedy_f1(pred_quads, gold_quads, _soft_match)

        return (hard_f1 + soft_f1) / 2
    except Exception:
        logger.warning("Exception in hate_recognition_reward_fn", exc_info=True)
        return 0.0
