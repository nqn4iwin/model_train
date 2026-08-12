"""이미 끝난 실험을 **저장된 채점 기록만으로** 다시 매긴다. GPU도 모델도 안 쓴다.

`baseline.py`가 실험마다 `eval/records.jsonl`에 건별로 남긴 것이 있다 -- 모델이 낸
판정, 교사 판정, AM 다섯 항목, 원문 출력까지. **판정 규칙이 바뀌었을 때 다시 학습할
이유가 없는 것은 이 파일 때문이다.** 다시 돌려도 같은 숫자가 나오고, 바뀌는 것은
그 숫자에 붙는 이름표뿐이다.

2026-08-12에 `scoring.collapsed`의 구멍을 고치면서 만들었다. 파싱 실패(`""`)를 판정
한 종류로 세는 바람에 `{'': 1, 'negative': 36}`이 붕괴 검사를 빠져나갔고, 실험 여섯이
잘못된 `됨` 판정을 받았다. 자세한 것은 그 함수의 설명에 있다.

**AM 값은 다시 계산해도 같아야 한다.** 달라지면 채점 규칙이 조용히 바뀐 것이므로
경고를 낸다 -- 라운드끼리 비교가 끊기는 종류의 사고다.

사용:
    python rescore.py              무엇이 바뀌는지 보여만 준다
    python rescore.py --write      summary.json에 실제로 반영한다
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoring import KEYS, collapsed, skew, verdict
from sweep import write_table
from train import read_config

HERE = Path(__file__).resolve().parent


def row_for(name: str, summary: dict, prior: dict) -> dict:
    """`sweep.py`가 만드는 것과 같은 모양의 표 한 줄. 설정에서 메모와 방식 이름을 붙인다."""
    path = HERE / "configs" / f"{name}.json"
    config = read_config(path) if path.exists() else {}
    return {"stage": "끝", "verdict": summary["verdict"],
            "note": config.get("note", ""),
            "peft_type": config.get("peft", {}).get("peft_type", "-"),
            **summary["AM_rates"], "평균": summary["AM_mean"],
            "교사일치": summary.get("teacher_agreement"),
            "쏠림": summary.get("skew"),
            "판정": summary.get("judgements"),
            "안 멈춤": summary.get("rambled_outputs"),
            # 걸린 시간은 기록에 안 남아 있다. 전에 만든 표에 있으면 그것을 쓴다.
            "분": prior.get("분", "-")}


def regrade(records: list[dict]) -> dict:
    """건별 기록에서 요약 값을 다시 만든다. `baseline.py`의 집계와 같은 식이다."""
    rates = {k: round(sum(r["scores"][k] for r in records) / len(records), 3)
             for k in KEYS}
    said = [r.get("judgement", "") for r in records]
    matched = sum(1 for r in records if r.get("judgement") == r.get("teacher_judgement"))
    is_collapsed = collapsed(said)
    return {
        "AM_rates": rates,
        "AM_mean": round(sum(rates.values()) / len(rates), 3),
        "AM_min": min(rates.values()),
        "verdict": "붕괴" if is_collapsed else verdict(rates),
        "collapsed": is_collapsed,
        "skew": skew(said),
        "teacher_agreement": round(matched / len(records), 3),
        "judgements": {j: said.count(j) for j in set(said)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--write", action="store_true",
                    help="summary.json을 실제로 고친다. 기본은 보여주기만 한다")
    args = ap.parse_args()

    # 전에 만든 표를 바탕으로 삼는다. **`못 돌림` 줄은 기록이 없어서 여기서만 나온다** --
    # 새로 짓겠다고 버리면 학습 자체가 실패한 열 개가 표에서 사라진다.
    table_path = args.runs / "sweep.json"
    table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else {}

    changed, same, broken = [], 0, []
    for records_path in sorted(args.runs.glob("*/eval/records.jsonl")):
        name = records_path.parents[1].name
        summary_path = records_path.parent / "summary.json"
        records = [json.loads(line) for line
                   in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            broken.append((name, "records.jsonl이 비었습니다"))
            continue

        fresh = regrade(records)
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        # AM 값이 달라지면 채점 규칙이 바뀐 것이다. 판정 이름표만 고치려던 작업이
        # 점수까지 건드렸다는 뜻이라, 조용히 넘기면 안 된다.
        if old.get("AM_rates") and old["AM_rates"] != fresh["AM_rates"]:
            broken.append((name, f"AM이 달라졌습니다 {old['AM_rates']} -> {fresh['AM_rates']}"))

        if old.get("verdict") == fresh["verdict"]:
            same += 1
        else:
            changed.append((name, old.get("verdict", "-"), fresh["verdict"],
                            fresh["teacher_agreement"], fresh["skew"], fresh["judgements"]))

        merged = {**old, **fresh}
        table[name] = row_for(name, merged, table.get(name, {}))
        if args.write:
            summary_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if changed:
        print(f"판정이 바뀐 것 {len(changed)}개\n")
        print(f"  {'실험':<28} {'전':<6} {'후':<6} {'교사일치':>8} {'쏠림':>7}  판정 분포")
        for name, before, after, agree, tilt, dist in changed:
            print(f"  {name:<28} {before:<6} {after:<6} {agree:>7.1%} "
                  f"{(tilt or 0):>6.1%}  {dist}")
    print(f"\n그대로인 것 {same}개")

    if broken:
        print(f"\n! 확인이 필요한 것 {len(broken)}개")
        for name, why in broken:
            print(f"  {name:<28} {why}")

    if args.write:
        write_table(table)
        print(f"\n표를 다시 썼습니다: runs/sweep.md · runs/sweep.json ({len(table)}줄)")
    else:
        print("\n보여주기만 했습니다. 반영하려면 --write 를 붙이세요.")
        print("**`sweep.py --all`은 부르지 마세요** -- 결과가 없는 설정을 아직 안 돌린 것으로")
        print("보고 다시 학습시킵니다. `못 돌림` 열 개가 전부 다시 돌아갑니다.")


if __name__ == "__main__":
    main()
