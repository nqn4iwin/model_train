"""few-shot 예시 선택이 고정되고 홀드아웃을 침범하지 않는지 확인한다."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from sft.formatting import build_fewshot_prefix, select_fewshot

ROOT = Path(__file__).resolve().parents[1]


class FewshotTest(unittest.TestCase):
    def test_zero_shots_is_empty(self) -> None:
        self.assertEqual(select_fewshot([], 0), [])
        self.assertEqual(build_fewshot_prefix([]), "")

    def test_first_positive_and_negative_are_kept(self) -> None:
        rows = [
            {"id": "n1", "judgement": "negative"},
            {"id": "n2", "judgement": "negative"},
            {"id": "p1", "judgement": "positive"},
        ]
        self.assertEqual([row["id"] for row in select_fewshot(rows, 3)], ["p1", "n1", "n2"])

    def test_fixed_examples_do_not_overlap_holdout(self) -> None:
        train = [json.loads(line) for line in
                 (ROOT / "data/20260821__annotate__v2.2-run2A/train.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        holdout = [json.loads(line) for line in
                   (ROOT / "data/20260821__annotate__v2.2-run2A/holdout.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        self.assertFalse({row["id"] for row in select_fewshot(train, 3)} &
                         {row["id"] for row in holdout})


if __name__ == "__main__":
    unittest.main()
