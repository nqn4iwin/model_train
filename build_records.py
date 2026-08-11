"""교사 해석(`records.jsonl`)에서 쓸 것만 거르고 학습용·평가용으로 가른다.

`data_collect`가 실제 문서에서 뽑은 실질 변경에 교사 모델(Solar)의 해석을 붙여 놓은
것이 입력이다. 여기서 하는 일은 **버릴 것을 버리고, 평가용을 갈라내는 것** 둘뿐이다.

**프롬프트와 정답 문자열은 여기서 만들지 않는다.** 규칙서를 붙일지, 정답을 어디까지
둘지는 실험 조건이라 `formatting.py`가 학습·평가 시점에 조립한다. 데이터를 한 벌로
얼려두어야 두 실험이 정말 같은 데이터를 썼는지 파일 해시로 확인된다.

**negative를 기본으로 남긴다.** 피어 세션은 빼라고 했지만 기획서 3.3 축 2가
'Negative 사례 제거'를 실험 조건으로 적어두었다 -- 처음부터 빼면 그 실험을 못 한다.
채점 항목 AM6s가 재는 것도 정확히 "negative일 때 입을 다무는가"인데, negative를 한 건도
안 본 모델이 그것을 할 리가 없다. 설계_메모 2절이 `impacts`에 대해 쓴 것과 같은
논리다 -- **넣어두면 나중에 뺄 수 있지만, 안 넣어두면 만들 수 없다.**

사용:
    python build_records.py <records.jsonl 경로> --out data/20260811__annotate__v2.2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# 평가용으로 통째로 빼두는 계열. 학습에 쓴 것으로 채점하면 점수가 부풀려진다. 해양수산
# 운영규정을 고른 이유는 37건뿐이라 학습 손해가 작으면서, 처리방침과 문체·형식이 전혀
# 달라 "처음 보는 문서에서도 되는가"를 실제로 재기 때문이다.
HOLDOUT_SERIES = "mof_rd_regulation_pair"

# 프롬프트를 깎을 때 쓴 11건. 학습에 들어가면 그 프롬프트로 잰 값이 무의미해진다.
# `evalset.json`은 `data_collect`에 있고 이 저장소에는 없으므로 걸러낼 `before_id`만
# 옮겨 적어 둔다. 11개뿐이고 늘어날 것이 아니라 파일을 통째로 들고 올 이유가 없다.
EVALSET_BEFORE_IDS = {
    "before_2022-B0618", "before_2022-B0817", "2020_standard_terms.converted-B0056",
    "mobile_2017.converted-B0088", "privacyOld14-B0885", "privacyOld14-B0117",
    "before_2022-B2063", "privacyOld14-B0870", "privacyOld11-B0930",
    "before_2026-40-B0003", "privacyOld12-B0610",
}

# 학습·평가에 실제로 쓰는 칸만 남긴다. 교사의 `raw`(원문 응답)와 `scores`는 이 저장소
# 에서 쓸 일이 없고, 셋을 다 두면 파일이 세 배가 된다.
FIELDS = ("id", "series", "before_id", "after_id", "before", "after",
          "judgement", "labels", "impacts", "direct_impact")


def usable(record: dict) -> bool:
    """학습 레코드가 될 수 있는가.

    `annotate.py`가 막 뱉은 것과, `data_collect`가 품질 필터까지 걸어 넘긴 것 두 가지가
    들어온다. **뒤엣것에는 `scores`가 없다** -- 이미 걸러진 뒤라 채점 결과를 들고 다닐
    이유가 없기 때문이다. `scores`가 있으면 여기서 한 번 더 거르고, 없으면 걸러진
    것으로 보고 판정만 확인한다.

    교사 호출이 실패했거나(`error`), 교사 출력이 JSON으로 안 읽힌 것(`AM1`이 0)은
    정답이 없으므로 쓸 수 없다.
    """
    if "error" in record:
        return False
    if "scores" in record and not record["scores"].get("AM1"):
        return False
    return record.get("judgement") in {"positive", "negative"}


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

    train, holdout, dropped = [], [], Counter()
    for record in records:
        if not usable(record):
            dropped["교사 출력을 못 씀"] += 1
            continue
        if record["before_id"] in EVALSET_BEFORE_IDS:
            dropped["평가셋 11건과 겹침"] += 1
            continue
        slim = {k: record.get(k) for k in FIELDS}
        (holdout if record["series"] == HOLDOUT_SERIES else train).append(slim)

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
