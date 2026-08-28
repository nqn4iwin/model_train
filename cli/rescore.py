"""이미 끝난 실험을 **저장된 채점 기록만으로** 다시 매긴다. GPU도 모델도 안 쓴다.

`cli/evaluate.py`가 실험마다 `eval/records.jsonl`에 건별로 남긴 것이 있다 -- 모델이 낸
판정, 교사 판정, AM 다섯 항목, 원문 출력까지. **판정 규칙이 바뀌었을 때 다시 학습할
이유가 없는 것은 이 파일 때문이다.** 다시 돌려도 같은 숫자가 나오고, 바뀌는 것은
그 숫자에 붙는 이름표뿐이다.

2026-08-12에 `scoring.collapsed`의 구멍을 고치면서 만들었다. 파싱 실패(`""`)를 판정
한 종류로 세는 바람에 `{'': 1, 'negative': 36}`이 붕괴 검사를 빠져나갔고, 실험 여섯이
잘못된 `됨` 판정을 받았다. 자세한 것은 그 함수의 설명에 있다.

**AM 값은 다시 계산해도 같아야 한다.** 달라지면 채점 규칙이 조용히 바뀐 것이므로
경고를 낸다 -- 라운드끼리 비교가 끊기는 종류의 사고다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.rescore              무엇이 바뀌는지 보여만 준다
    python -m cli.rescore --write      summary.json을 고치고 표를 다시 쓴다
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli.sweep import table_stem, write_table
from sft.records import ruler_name
from sft.scoring import (KEYS, collapsed, label_agreement, layered_agreement,
                        score_blind, skew, verdict)
from sft.training import read_config

# cli/ 안에 있으므로 저장소 뿌리는 한 단계 위다.
ROOT = Path(__file__).resolve().parents[1]

# 교사 라벨을 여기서 가져온다. **`records.jsonl`에는 교사 판정만 있고 교사 라벨이 없다** --
# 옛 실험 100여 개가 그렇다. 그래서 `id`로 홀드아웃과 맞붙인다. `cli/sweep.py:37`과 같은
# 얼려둔 파일이라 어느 라운드를 다시 매기든 같은 37건 위에서 이뤄진다.
HOLDOUT = "data/20260811__annotate__v2.2/holdout.jsonl"


def row_for(name: str, summary: dict, prior: dict,
            config_name: str | None = None) -> dict:
    """`sweep.py`가 만드는 것과 같은 모양의 표 한 줄. 설정에서 메모와 방식 이름을 붙인다.

    `config_name`은 **설정 파일을 찾을 이름**이다. 보통 줄 이름과 같지만, 같은 어댑터를
    다른 조건으로 또 채점한 변종 줄(`...-3shot`)은 설정 파일이 따로 없으므로 원래 실험
    이름을 넘긴다. 안 넘기면 `note`와 `peft_type`이 빈칸이 되어 **표에서 그 줄만 어떤
    학습법인지 안 보인다.**
    """
    path = ROOT / "configs" / f"{config_name or name}.json"
    config = read_config(path) if path.exists() else {}
    # 변종 줄은 설정의 메모를 그대로 물려받는다. 그러면 **두 줄이 똑같은 메모를 달고
    # 나란히 앉아** 어느 쪽이 무엇인지 표에서 안 보이므로, 꼬리를 앞에 박아 둔다.
    note = config.get("note", "")
    if config_name and config_name != name:
        note = f"[{name.removeprefix(config_name).lstrip('-')}] {note}".strip()
    return {"stage": "끝", "verdict": summary["verdict"],
            "note": note,
            "peft_type": config.get("peft", {}).get("peft_type", "-"),
            **summary["AM_rates"], "평균": summary["AM_mean"],
            "에폭": config.get("num_train_epochs"),
            "교사일치": summary.get("teacher_agreement"),
            "건진판정": summary.get("teacher_agreement_salvaged"),
            "라벨일치": summary.get("label_agreement"),
            "대상일치": summary.get("target_agreement"),
            "방향일치": summary.get("direction_agreement"),
            "쌍일치": summary.get("pair_agreement"),
            "못건짐": summary.get("label_unrecoverable"),
            "쏠림": summary.get("skew"),
            "판정": summary.get("judgements"),
            "안 멈춤": summary.get("rambled_outputs"),
            # 걸린 시간은 기록에 안 남아 있다. 전에 만든 표에 있으면 그것을 쓴다.
            "분": prior.get("분", "-")}


def reparse(records: list[dict], label_free: bool = False) -> tuple[list[dict], int]:
    """저장된 점수를 버리고 **원문(`raw`)에서 다시 매긴다.** GPU도 모델도 안 쓴다.

    **이 도구는 원래 저장된 점수를 다시 모으기만 한다**(`regrade`의 `r["scores"][k]`).
    그래서 `sft.scoring.parse_output`을 고쳐도 표에 안 닿는다 -- 2026-08-27에 파서를
    갈아 끼우고 `--write`를 돌렸더니 119줄이 전부 "그대로"로 나온 자리다.

    **`records.jsonl`은 안 고친다.** 거기 든 `raw`가 원본이고 점수는 그것에서 언제든
    다시 나온다. 파생값을 원본 파일에 덮어쓰면 다음에 파서를 또 고칠 때 되돌릴 곳이
    없어진다. 고쳐 쓰는 것은 `summary.json`과 표뿐이다.

    **`restatement_ratio`는 다시 안 잰다.** 그 값은 원문 블록이 있어야 나오는데 이
    도구는 홀드아웃에서 교사 라벨만 읽는다. 새로 건져 낸 문장은 그 칸이 `None`으로
    남는다 -- 합격 판정에 안 쓰는 값이라 표는 안 흔들린다.
    """
    fresh, moved = [], 0
    for record in records:
        raw = record.get("raw")
        if raw is None:          # 원문이 없는 옛 기록은 손대지 않는다
            fresh.append(record)
            continue
        scored = score_blind(raw, label_free)
        parsed = scored.pop("parsed") or {}
        if scored != record.get("scores"):
            moved += 1
        fresh.append({**record, "scores": scored,
                      "judgement": str(parsed.get("judgement", "") or ""),
                      "labels": parsed.get("labels"),
                      "impacts": parsed.get("impacts"),
                      "direct_impact": parsed.get("direct_impact")})
    return fresh, moved


def regrade(records: list[dict], teacher: dict[str, list] | None = None,
            label_free: bool = False) -> dict:
    """건별 기록에서 요약 값을 다시 만든다. `cli.evaluate`의 집계와 같은 식이다.

    `teacher`는 `{id: 교사 labels}`다. 기록 자체에 `teacher_labels`가 있으면 그것을 먼저
    쓰고, 없으면(옛 실험이 그렇다) 이 사전에서 `id`로 찾는다.
    """
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
        **_labels(records, teacher or {}, label_free),
        # 층별 열. **`teacher_agreement`·`label_agreement` 위에 덮지 않고 옆에 선다** --
        # 저 둘의 값이 바뀌면 지난 147줄 표와 비교가 끊긴다.
        **layered_agreement(records, teacher or {}, label_free),
    }


def _labels(records: list[dict], teacher: dict[str, list],
            label_free: bool = False) -> dict:
    """라벨일치를 요약 모양으로 만든다. 값이 없으면 `None`으로 둔다."""
    agreement = label_agreement([
        (r.get("labels"), r.get("teacher_labels", teacher.get(r.get("id"))))
        for r in records], label_free)
    return {"label_agreement": agreement["rate"],
            "label_matched": agreement["matched"],
            "label_denominator": agreement["denominator"],
            "label_note": agreement["note"]}


def read_teacher(path: Path) -> dict[str, list]:
    """홀드아웃에서 `{id: labels}`만 뽑는다."""
    teacher = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            teacher[row["id"]] = row.get("labels")
    return teacher


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--data", default=HOLDOUT,
                    help="교사 라벨을 가져올 홀드아웃. 얼려둔 것을 그대로 쓴다")
    ap.add_argument("--write", action="store_true",
                    help="summary.json을 실제로 고친다. 기본은 보여주기만 한다")
    ap.add_argument("--reparse", action="store_true",
                    help="저장된 점수 대신 원문에서 다시 매긴다."
                         " **파서를 고친 날에만 쓴다** -- AM 값이 움직인다")
    args = ap.parse_args()
    teacher = read_teacher(ROOT / args.data)
    # **`--data` 하나가 셋을 다 정한다** -- 교사 라벨을 어디서 가져올지, 어느 채점
    # 폴더를 읽을지, 어느 표에 쓸지. 셋이 따로 놀면 37건으로 잰 값과 135건으로 잰 값이
    # 한 표에 앉는다(2026-08-24에 실제로 그럴 뻔했다).
    folder = ruler_name(ROOT / args.data)
    stem = table_stem(folder)
    print(f"자: {args.data}  ({len(teacher)}건) -> runs/*/{folder}*/ -> runs/{stem}.md\n")

    # 전에 만든 표를 바탕으로 삼는다. **`못 돌림` 줄은 기록이 없어서 여기서만 나온다** --
    # 새로 짓겠다고 버리면 학습 자체가 실패한 열 개가 표에서 사라진다.
    table_path = args.runs / f"{stem}.json"
    table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else {}

    changed, same, broken = [], 0, []
    reparsed = 0
    # 홀드아웃 경로 -> 자 이름. 같은 파일을 채점 기록 수백 개마다 다시 읽지 않는다.
    rulers: dict[str, str] = {}

    for records_path in sorted(args.runs.glob("*/eval-*/records.jsonl")):
        eval_folder = records_path.parent.name
        run = records_path.parents[1].name
        summary_path = records_path.parent / "summary.json"
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        # **어느 자로 잰 값인지는 폴더 이름이 아니라 `summary.json`의 `data`가 정한다.**
        # 폴더 이름을 앞맞추기로 가르면 `eval-mof`가 `eval-mof-motie`까지 집어삼켜
        # **37건 자 표에 135건 자 값이 앉는다** -- 2026-08-24에 겪은 것과 같은 종류의
        # 사고다. `data`에는 그 채점이 실제로 읽은 홀드아웃 경로가 적혀 있고, 채점
        # 기록 511개에 하나도 빠짐없이 들어 있다(2026-08-26 확인).
        #
        # 이렇게 두면 **채점 변종이 새로 생겨도 코드를 안 고친다.** 폴더 이름이
        # `eval-mof-motie-3shot`이든 `-e1`이든, 같은 자로 쟀으면 같은 표에 앉는다.
        used = old.get("data")
        if used:
            if used not in rulers:
                rulers[used] = ruler_name(ROOT / used)
            if rulers[used] != folder:
                continue
        elif eval_folder != folder:
            continue

        # 자 이름 뒤에 남은 꼬리가 채점 변종이다 -- `eval-mof-motie-3shot` -> `-3shot`.
        # 표에서 줄을 가르는 데만 쓰고, 설정 파일은 실험 폴더 이름으로 찾는다.
        name = run + eval_folder.removeprefix(folder)
        records = [json.loads(line) for line
                   in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            broken.append((name, "records.jsonl이 비었습니다"))
            continue

        # **홀드아웃 건수보다 적으면 실험이 아니라 채점 찌꺼기다.** 2026-08-18에
        # `runs/frod/eval/`에 한 줄짜리가 남아 있어, 표에서 `못 돌림`이던 1차 frod가
        # 1건짜리 `됨`으로 뒤집힐 뻔했다. 여기서 막지 않으면 옛 라운드의 기록이
        # 조용히 거짓이 된다. 반복 채점(`--repeat`)은 건수가 늘어나므로 안 걸린다.
        if teacher and len(records) < len(teacher):
            broken.append((name, f"{len(records)}건뿐입니다 (홀드아웃 {len(teacher)}건)"
                                 " -- 채점 찌꺼기로 보고 건너뜁니다"))
            continue

        # `target`은 **실제로 돌아간 설정**에서 읽는다. `cli/train.py`가 실험 폴더에
        # `config.json`을 남기므로, `configs/`에 파일이 없는 기준 조건도 여기서 나온다.
        ran = records_path.parents[1] / "config.json"
        target = (json.loads(ran.read_text(encoding="utf-8")).get("target")
                  if ran.exists() else None)
        moved = 0
        if args.reparse:
            records, moved = reparse(records, label_free=(target == "sentence"))
            reparsed += moved
        fresh = regrade(records, teacher, label_free=(target == "sentence"))

        # AM 값이 달라지면 채점 규칙이 바뀐 것이다. 판정 이름표만 고치려던 작업이
        # 점수까지 건드렸다는 뜻이라, 조용히 넘기면 안 된다.
        #
        # **`--reparse`일 때는 달라지는 것이 목적이다.** 그때는 사고가 아니라 몇 건이
        # 움직였는지를 세어 맨 끝에 한 줄로 알린다. 그래도 검사를 끄지는 않는다 --
        # 원문을 다시 안 읽었는데도 값이 달라졌다면 그건 여전히 사고다.
        if old.get("AM_rates") and old["AM_rates"] != fresh["AM_rates"] and not args.reparse:
            broken.append((name, f"AM이 달라졌습니다 {old['AM_rates']} -> {fresh['AM_rates']}"))

        if old.get("verdict") == fresh["verdict"]:
            same += 1
        else:
            changed.append((name, old.get("verdict", "-"), fresh["verdict"],
                            fresh["teacher_agreement"], fresh["skew"], fresh["judgements"]))

        merged = {**old, **fresh}
        table[name] = row_for(name, merged, table.get(name, {}), config_name=run)
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
    if args.reparse:
        print(f"원문에서 다시 매겨 AM이 움직인 기록 {reparsed}건"
              " -- **이 표의 AM 값은 새 파서의 값이라 옛 표와 나란히 놓으면 안 됩니다.**")

    if broken:
        print(f"\n! 확인이 필요한 것 {len(broken)}개")
        for name, why in broken:
            print(f"  {name:<28} {why}")
        # **AM이 달라졌다는 것이 줄줄이 나오면 십중팔구 `--reparse`를 빼먹은 것이다.**
        # `records.jsonl`에 저장된 `scores`는 그 채점을 돌리던 날의 파서로 매긴 값이고,
        # 2026-08-27에 `parse_output`이 바뀌었다. 그대로 `--write` 하면 표가 옛 값으로
        # 되돌아간다. 되돌린 표는 `--reparse --write`로 다시 지으면 살아난다.
        if not args.reparse and any("AM이 달라졌습니다" in why for _, why in broken):
            print("\n  ** `--reparse`를 안 붙이셨습니다. **"
                  " 저장된 점수는 옛 파서의 값이라 표가 되돌아갑니다.")

    if args.write:
        write_table(table, stem, args.data)
        print(f"\n표를 다시 썼습니다: runs/{stem}.md · runs/{stem}.json ({len(table)}줄)")
    else:
        print("\n보여주기만 했습니다. 반영하려면 --write 를 붙이세요.")
        print("**`cli.sweep --all`은 부르지 마세요** -- 결과가 없는 설정을 아직 안 돌린")
        print("것으로 보고 다시 학습시킵니다. `못 돌림` 열 개가 전부 다시 돌아갑니다.")


if __name__ == "__main__":
    main()
