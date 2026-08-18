"""교사 해석(`records.jsonl`)에서 쓸 것만 거르고 학습용·평가용으로 가른다.

거르기와 가르기 규칙 자체는 `sft/records.py`에 있다. 여기는 **파일을 읽고 쓰고
무엇이 얼마나 버려졌는지 사람에게 보여주는 것**만 한다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.build_records <records.jsonl 경로> --out data/20260811__annotate__v2.2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sft.records import HOLDOUT_SERIES, split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", help="data_collect가 뽑은 records.jsonl")
    ap.add_argument("--out", required=True, help="쓸 폴더. 날짜와 판본을 경로에 박는다")
    ap.add_argument("--train-only", action="store_true",
                    help="이미 얼려둔 holdout.jsonl을 건드리지 않는다. baseline을 그 위에서 "
                         "이미 쟀다면 반드시 이것을 준다 -- 홀드아웃이 한 건이라도 달라지면 "
                         "학습 전과 후를 나란히 놓을 수 없다")
    args = ap.parse_args()

    source = Path(args.records)
    records = [json.loads(line) for line in
               source.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, holdout, dropped = split(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [("train.jsonl", train)]
    if args.train_only:
        frozen = out_dir / "holdout.jsonl"
        holdout = [json.loads(line) for line in
                   frozen.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"홀드아웃은 얼려둔 {frozen}을 그대로 씁니다 ({len(holdout)}건)\n")
    else:
        written.append(("holdout.jsonl", holdout))
    for name, chosen in written:
        (out_dir / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen),
            encoding="utf-8")

    meta = {
        "source": source.name, "source_run": source.parent.name,
        "source_records": len(records),
        "holdout_frozen": args.train_only,
        "holdout_series": HOLDOUT_SERIES,
        "train": len(train), "holdout": len(holdout),
        "train_judgements": dict(Counter(r["judgement"] for r in train)),
        "train_series": dict(Counter(r["series"] for r in train)),
        "holdout_judgements": dict(Counter(r["judgement"] for r in holdout)),
        "dropped": dict(dropped),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"원본 {len(records)}건")
    for reason, count in dropped.most_common():
        print(f"  버림  {reason:<24} {count}건")
    print(f"\n  학습    {len(train):>4}건  {meta['train_judgements']}")
    for name, count in Counter(r["series"] for r in train).most_common():
        print(f"            {name:<34}{count:>4}건")
    print(f"  홀드아웃 {len(holdout):>4}건  {meta['holdout_judgements']}  ({HOLDOUT_SERIES})")
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
