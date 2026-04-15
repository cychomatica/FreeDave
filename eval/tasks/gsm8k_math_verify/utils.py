"""GSM8K: exact_match (flexible-extract style) + math_verify on full completion."""

import logging
import re

eval_logger = logging.getLogger(__name__)

# Same pattern / index as lm_eval gsm8k ``flexible-extract`` filter.
_GSM8K_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _flexible_extract(pred: str) -> str:
    if not isinstance(pred, str):
        pred = ""
    matches = _GSM8K_FLEXIBLE_RE.findall(pred)
    if not matches:
        return ""
    match = matches[-1]
    if isinstance(match, tuple):
        non_empty = [m for m in match if m]
        if not non_empty:
            return ""
        match = non_empty[0]
    return str(match).strip()


def _gsm8k_gold_numeric(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()


def score_gsm8k_completion(doc: dict, completion: str) -> dict[str, int]:
    """
    Same scoring as ``process_results`` for one example: ``doc`` is GSM8K row
    (must have ``answer``); ``completion`` is the model string (full generation).

    Usable from offline scripts that replay ``samples_*.jsonl`` without lm_eval.
    """
    gold_answer_field = doc["answer"]

    pred_extracted = _flexible_extract(completion)
    try:
        import numpy as np
        from lm_eval.api.metrics import exact_match_hf_evaluate

        em = exact_match_hf_evaluate(
            predictions=np.array([pred_extracted]),
            references=np.array([gold_answer_field]),
            regexes_to_ignore=[",", r"\$", r"(?s).*#### ", r"\.$"],
            ignore_case=True,
            ignore_punctuation=False,
        )["exact_match"]
        exact_match = int(bool(em))
    except Exception as e:
        eval_logger.debug("exact_match (lm_eval) failed: %s", e)
        ref = gold_answer_field
        for pat in [r"(?s).*#### ", ",", r"\$", r"\.$"]:
            ref = re.sub(pat, "", ref)
        pred = pred_extracted
        for pat in [",", r"\$", r"\.$"]:
            pred = re.sub(pat, "", pred)
        exact_match = int(ref.strip().lower() == pred.strip().lower())

    gold = _gsm8k_gold_numeric(gold_answer_field)
    try:
        from math_verify import parse, verify

        ok = bool(verify(gold=parse(gold), target=parse(completion)))
    except Exception as e:
        eval_logger.debug("math_verify failed: %s", e)
        ok = False

    return {"exact_match": exact_match, "math_verify": int(ok)}


def process_results(doc: dict, results: list) -> dict[str, int]:
    return score_gsm8k_completion(doc, results[0])
