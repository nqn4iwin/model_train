"""LLM 심판(코덱스)에게 물을 거리를 만든다. **GPU도 네트워크도 안 쓴다.**

**두 가지를 따로 묻는다. 섞으면 뭐가 틀렸는지 못 가른다.**

    Q1  개정 전과 후에 **유의미한 변경**이 있나
        -> 교사의 `judgement` 자체를 검사한다. **교사 답을 안 보여준다**
        -> 2026-08-25에 사람이 36장 읽어 **negative여야 하는데 positive로 달린 것 6건**을
           찾았다. 나머지 99건에 몇 개가 더 있는지 아무도 모른다

    Q2  교사 서술과 학생 서술이 **같은 내용**인가
        -> **라벨을 안 보여준다.** 이 저장소가 오래 주장해온
           「라벨이 달라도 서술은 얼추 같다」를 실제로 재는 자리다

**Q2는 라벨이 같은 건과 다른 건을 갈라서 봐야 뜻이 있다.** 둘 다 「같다」가 높게 나오면
**라벨일치가 서술을 예측하지 못한다**는 것이 실측으로 확인된다 -- 그것이 이 문서가
`라벨일치를 품질 대리값으로 쓰지 않는다`고 적어둔 근거가 된다.

**심판은 판정하지 않는다. 후보만 낸다.** `verdict`도 표도 안 건드린다. 사람이 볼 목록을
좁히는 데만 쓴다 -- 「열만 늘리고 판정은 사람이 읽는다」와 같은 자리다.

**답은 파일로 얼린다.** `cli.rescore`는 GPU도 네트워크도 없이 표를 다시 짓는 것이 존재
이유인데, LLM은 같은 질문에 다른 답을 낼 수 있다. 심판 답을 JSONL로 남겨 **두 번째부터는
그 파일만 읽는다.**

**겹치기를 넣는다.** 같은 건을 두 번 물어 갈리는 비율을 먼저 잰다 -- 2026-08-25에
사람 판정에서 겹친 4장이 4장 다 갈렸다.

사용 (**저장소 뿌리에서 `-m`으로**):

    python -m H_eval.build_judge --run runs/delora-run2A --out H_eval/20260825__judge
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from sft.scoring import label_pairs

HOLDOUT = "data/20260821__annotate__v2.2-run2A/holdout.jsonl"

# 규칙서 1단계를 그대로 옮긴다. **`prompts/roleA_v2.2.txt`가 원본이고 여기가 사본이다** --
# 원본이 바뀌면 여기도 손으로 맞춘다. 심판에게 전문을 주면 라벨 어휘까지 배워
# "무엇이 바뀌었나"로 넘어가 버리므로 1단계만 준다.
STAGE1 = """판단 기준은 **글자가 달라졌는지가 아니라, 이 문서가 규정하는 권리·의무·조건·절차·
적용 범위 중 하나라도 달라졌는지**입니다.

아래는 그 기준을 적용한 예입니다. 목록에 없는 경우에도 위 기준으로 판단하세요.

- 다른 조항이 신설·삭제되어 조·항·호 번호가 밀린 것 (제27조 → 제28조)
- 다른 조항을 가리키는 인용 번호가 그 밀림을 따라 바뀐 것
- 항목기호, 띄어쓰기, 문장부호, 같은 뜻의 표기 차이
- 문서 자체의 제·개정일자, 시행일, 판번호 표기
    이 문서가 언제부터 유효한지를 밝히는 표기입니다. 조항이 요구하는 기한이 아닙니다.

이것들은 숫자나 날짜가 바뀌지만 **조항이 요구하는 내용이 하나도 달라지지 않았습니다.**"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_q1(holdout: list[dict], rnd: random.Random, dup: int) -> list[dict]:
    """개정 전후에 유의미한 변경이 있나. **교사 답은 안 넣는다.**"""
    items = [{"qid": f"q1-{i:04d}", "id": r["id"],
              "before": r["before"], "after": r["after"]}
             for i, r in enumerate(holdout)]
    # 겹치기. 같은 건을 다른 qid로 한 번 더 넣는다 -- 심판이 눈치채면 안 된다.
    for j, src in enumerate(rnd.sample(items, min(dup, len(items)))):
        items.append({**src, "qid": f"q1-dup{j:03d}"})
    rnd.shuffle(items)
    return items


def build_q2(holdout: list[dict], records: list[dict], rnd: random.Random,
             dup: int, tag: str) -> tuple[list[dict], dict]:
    """교사 서술과 학생 서술이 같은 내용인가. **라벨을 안 넣는다.**

    `key`에 어느 쪽이 교사였는지와 **라벨이 같았는지**를 따로 남긴다 -- 심판은 못 보고
    집계할 때만 쓴다.
    """
    by_id = {r["id"]: r for r in holdout}
    items, key = [], {}
    for i, rec in enumerate(records):
        row = by_id.get(rec["id"])
        if row is None:
            continue
        theirs, mine = row.get("direct_impact", ""), rec.get("direct_impact", "")
        if not theirs or not mine:
            continue          # 한쪽이라도 문장이 없으면 견줄 것이 없다
        flip = rnd.random() < 0.5
        qid = f"q2-{tag}-{i:04d}"
        items.append({"qid": qid, "id": rec["id"],
                      "before": row["before"], "after": row["after"],
                      "갑": mine if flip else theirs,
                      "을": theirs if flip else mine})
        tp, mp = label_pairs(row.get("labels")), label_pairs(rec.get("labels"))
        key[qid] = {"id": rec["id"], "갑이_학생": flip,
                    "라벨같음": (set(tp or []) == set(mp or []))}
    for j, src in enumerate(rnd.sample(items, min(dup, len(items)))):
        qid = f"q2-{tag}-dup{j:03d}"
        items.append({**src, "qid": qid})
        key[qid] = {**key[src["qid"]], "겹치기_원본": src["qid"]}
    rnd.shuffle(items)
    return items, key


TASK = """# 심판 과제 — 읽고 그대로 따른다

**너는 채점자가 아니라 후보를 골라 주는 사람이다.** 여기서 나온 답은 표에 안 들어가고,
사람이 무엇을 먼저 볼지 정하는 데만 쓴다. **모르겠으면 모르겠다고 하는 편이 낫다.**

## 공통 규칙

- 입력 한 줄이 물음 하나다. **`qid`를 그대로 답에 옮긴다.**
- **25줄씩 끊어서** 처리하고 그때마다 출력 파일에 이어 붙인다. 한 번에 다 하지 않는다.
- 답은 **JSONL 한 줄에 하나**다. 설명 문장이나 코드펜스를 앞뒤에 붙이지 않는다.
- **같은 `id`가 두 번 나올 수 있다.** 일부러 넣은 것이니 앞에 뭐라 답했는지 찾지 말고
  그 줄만 보고 답한다.

---

## Q1 — `q1.jsonl` -> `q1_답.jsonl`

**묻는 것: 개정 전과 후 사이에 유의미한 변경이 있는가.**

{stage1}

**`before`와 `after`만 보고 판단한다.** 그 밖의 정보는 주지 않는다.

출력 한 줄:

    {{"qid": "q1-0007", "유의미": true, "확신": "높음", "근거": "제출 기한이 30일에서 60일로 바뀌어 의무의 내용이 달라졌다"}}

- `유의미` : `true` = 권리·의무·조건·절차·적용 범위 중 하나가 달라졌다 · `false` = 안 달라졌다
- `확신` : `높음` · `보통` · `낮음`. **애매하면 낮음을 쓴다.** 낮음이 곧 사람이 볼 목록이다
- `근거` : 한 문장. 어디가 달라졌는지(또는 왜 안 달라졌는지)

---

## Q2 — `q2.jsonl` -> `q2_답.jsonl`

**묻는 것: 「갑」과 「을」 두 서술이 같은 내용인가.**

둘 다 같은 개정(`before` -> `after`)을 설명한 문장이다. **어느 쪽이 누가 쓴 것인지는
알려주지 않고, 알 필요도 없다.** 잘 썼는지 · 읽기 좋은지 · 어느 쪽이 나은지도 묻지 않는다.
**오직 말하는 내용이 같은가만 본다.**

- 표현이 달라도 **같은 사실을 말하면 「같다」**이다. 이것이 이 물음의 핵심이다
- 한쪽이 더 자세한 것은 **그 자체로는 다름이 아니다.** 자세한 쪽이 더 말한 내용이
  다른 쪽과 어긋나지 않으면 「같다」, 빠진 것이 중요하면 「일부」
- **한쪽이 사실을 틀리게 말했으면 「다르다」**이다

출력 한 줄:

    {{"qid": "q2-0031", "같은내용": "같다", "확신": "높음", "근거": "둘 다 심의 대상이 좁아진 것을 말한다. 갑은 예외 사유를 더 적었을 뿐이다"}}

- `같은내용` : `같다` · `일부` · `다르다`
- `확신` : `높음` · `보통` · `낮음`
- `근거` : 한 문장. **다르다면 무엇이 어긋나는지**를 반드시 적는다

---

## 다 끝나면

두 파일의 줄 수가 입력과 맞는지 세어 보고 알려준다. 집계는 하지 않는다 --
`python -m H_eval.read_judge`가 한다.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 심판에게 물을 거리를 만든다")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--run", action="append", metavar="DIR",
                    help="Q2를 만들 실행 폴더. 여러 번 줄 수 있다")
    ap.add_argument("--eval-dir", default="eval-mof-motie")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dup", type=int, default=10, help="겹치기로 더 넣을 건수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    holdout = read_jsonl(Path(args.holdout))
    args.out.mkdir(parents=True, exist_ok=True)

    q1 = build_q1(holdout, rnd, args.dup)
    (args.out / "q1.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in q1), encoding="utf-8")

    q2, key = [], {}
    for run in (args.run or []):
        path = Path(run) / args.eval_dir / "records.jsonl"
        if not path.exists():
            print(f"  건너뜀 (없음): {path}")
            continue
        items, k = build_q2(holdout, read_jsonl(path), rnd, args.dup, Path(run).name)
        for x in items:
            x["run"] = Path(run).name
        for qid, v in k.items():
            v["run"] = Path(run).name
        q2 += items
        key |= k
    if q2:
        (args.out / "q2.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in q2), encoding="utf-8")
        (args.out / "q2_열쇠.json").write_text(
            json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    (args.out / "과제.md").write_text(
        TASK.format(stage1=textwrap.indent(STAGE1, "")), encoding="utf-8")

    same = sum(1 for v in key.values() if v["라벨같음"])
    print(f"저장: {args.out}")
    print(f"  q1.jsonl  {len(q1):4d}줄  (홀드아웃 {len(holdout)} + 겹치기 {args.dup})")
    print(f"  q2.jsonl  {len(q2):4d}줄  (라벨같음 {same} · 라벨다름 {len(key) - same})")
    print(f"  과제.md   코덱스에게 이 파일을 읽히고 시작한다")
    print("\n  ! q2_열쇠.json 은 심판에게 주지 않는다 — 어느 쪽이 교사인지가 들어 있다")


if __name__ == "__main__":
    main()
