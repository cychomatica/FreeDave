#!/usr/bin/env python3
"""
Replay gsm8k_math_verify metrics on an existing lm_eval ``samples_gsm8k_*.jsonl``
without re-running the model. Skips duplicate rows (same doc_id, different filter).

Usage (from repo root or eval/):
  python eval/tasks/gsm8k_math_verify/rescore_local_file.py <local_jsonl_file_path>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _stderr_mean(n: int, p: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p) / n, 0.0))


def main() -> None:
    try:
        import math_verify  # noqa: F401
    except ImportError:
        print(
            "Warning: `math_verify` not installed; math_verify scores will be 0. "
            "Install: pip install math_verify sympy antlr4-python3-runtime==4.11",
            file=sys.stderr,
        )

    parser = argparse.ArgumentParser(description="Rescore GSM8K samples with gsm8k_math_verify logic.")
    parser.add_argument(
        "jsonl",
        nargs="+",
        help="One or more samples_gsm8k_*.jsonl paths (only first run per doc_id is used).",
    )
    parser.add_argument(
        "--write-jsonl",
        metavar="PATH",
        help="Optional path to write per-doc scores as JSONL (doc_id, exact_match, math_verify).",
    )
    args = parser.parse_args()

    eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(eval_dir, "tasks", "gsm8k_math_verify"))
    from utils import score_gsm8k_completion  # noqa: E402

    seen: set[int | str] = set()
    em_sum = 0
    mv_sum = 0
    n = 0
    out_lines: list[dict] = []

    for path in args.jsonl:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id")
                if doc_id in seen:
                    continue
                seen.add(doc_id)

                doc = row.get("doc") or {}
                if "answer" not in doc:
                    raise KeyError(f"missing doc.answer at doc_id={doc_id} in {path}")

                resps = row.get("resps") or []
                if not resps or not resps[0]:
                    raise ValueError(f"empty resps at doc_id={doc_id} in {path}")
                completion = resps[0][0] if isinstance(resps[0], list) else resps[0]

                scores = score_gsm8k_completion(doc, completion)
                em_sum += scores["exact_match"]
                mv_sum += scores["math_verify"]
                n += 1
                out_lines.append(
                    {
                        "doc_id": doc_id,
                        "source": path,
                        "exact_match": scores["exact_match"],
                        "math_verify": scores["math_verify"],
                    }
                )

    if n == 0:
        print("No samples found (empty files or no unique doc_id).", file=sys.stderr)
        sys.exit(1)

    p_em = em_sum / n
    p_mv = mv_sum / n
    print(f"examples: {n}")
    print(f"exact_match (flexible-style): {p_em:.6f} ± {_stderr_mean(n, p_em):.6f}")
    print(f"math_verify:                  {p_mv:.6f} ± {_stderr_mean(n, p_mv):.6f}")

    if args.write_jsonl:
        with open(args.write_jsonl, "w", encoding="utf-8") as out:
            for row in out_lines:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote per-example scores to {args.write_jsonl}")


if __name__ == "__main__":
    main()
