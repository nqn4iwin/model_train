"""심판이 낸 답을 읽어 집계한다. **GPU도 네트워크도 안 쓴다.**

`H_eval.build_judge`가 만든 물음과 심판의 답을 맞춰 본다. **표(`sweep*.md`)는 안 건드린다** --
여기서 나오는 것은 판정이 아니라 **사람이 볼 후보 목록**이다.

세 가지를 낸다.

    Q1  교사가 `positive`라 했는데 심판이 "안 달라졌다"고 한 건
        -> **정답키 의심 목록.** 2026-08-25에 사람이 36장 읽어 6건을 찾은 그 종류다

    Q2  `같은내용`을 **라벨이 같은 건과 다른 건으로 갈라** 센다
        -> 둘이 비슷하면 **라벨일치가 서술을 예측하지 못한다**가 실측이 된다

    겹치기  같은 것을 두 번 물어 답이 갈린 비율
        -> **이 값이 크면 위 두 숫자를 믿으면 안 된다.** 먼저 본다

사용 (**저장소 뿌리에서 `-m`으로**):

    python -m H_eval.read_judge --dir H_eval/20260825__judge
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import shlex
import sys
from pathlib import Path

HOLDOUT = "data/20260821__annotate__v2.2-run2A/holdout.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("```"):
                continue          # 심판이 코드펜스를 붙였을 때를 봐준다
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! {path.name} {n}번째 줄을 못 읽었다", file=sys.stderr)
    return out


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description="심판 답을 집계한다")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--out", type=Path, help="HTML 경로. 안 주면 화면에만")
    args = ap.parse_args()
    D = args.dir

    teacher = {}
    with Path(args.holdout).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            teacher[r["id"]] = r

    q1_ask = {x["qid"]: x for x in read_jsonl(D / "q1.jsonl")}
    q1_ans = {x["qid"]: x for x in read_jsonl(D / "q1_답.jsonl")}
    q2_ask = {x["qid"]: x for x in read_jsonl(D / "q2.jsonl")}
    q2_ans = {x["qid"]: x for x in read_jsonl(D / "q2_답.jsonl")}
    key = json.loads((D / "q2_열쇠.json").read_text(encoding="utf-8")) \
        if (D / "q2_열쇠.json").exists() else {}

    blocks, lines = [], []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text)

    # ── 겹치기부터. 이게 크면 아래 숫자를 믿으면 안 된다 ──────────────────────
    say("── 겹치기 (같은 것을 두 번 물었을 때 답이 갈린 비율)")
    for tag, ask, ans, field in (("Q1", q1_ask, q1_ans, "유의미"),
                                 ("Q2", q2_ask, q2_ans, "같은내용")):
        pairs = []
        by_id = collections.defaultdict(list)
        for qid in ask:
            if qid in ans:
                by_id[(ask[qid]["id"], ask[qid].get("run", ""))].append(qid)
        for qids in by_id.values():
            if len(qids) >= 2:
                vals = {json.dumps(ans[q].get(field), ensure_ascii=False) for q in qids}
                pairs.append(len(vals) > 1)
        if pairs:
            say(f"   {tag}  {len(pairs)}쌍 중 {sum(pairs)}쌍이 갈렸다  ({pct(sum(pairs), len(pairs))})")
        else:
            say(f"   {tag}  겹친 쌍이 없다")
    say()

    # ── Q1. 정답키 의심 목록 ────────────────────────────────────────────────
    suspect = []
    if q1_ans:
        cross = collections.Counter()
        for qid, a in q1_ans.items():
            ask_row = q1_ask.get(qid)
            if not ask_row or "dup" in qid:
                continue
            t = teacher.get(ask_row["id"], {}).get("judgement")
            j = a.get("유의미")
            cross[(t, bool(j))] += 1
            if t == "positive" and j is False:
                suspect.append({"id": ask_row["id"], "확신": a.get("확신", ""),
                                "근거": a.get("근거", "")})
        say("── Q1  교사 판정 대 심판")
        say(f"   {'교사':10s}{'심판: 달라졌다':>14s}{'심판: 안 달라졌다':>18s}")
        for t in ("positive", "negative"):
            say(f"   {t:10s}{cross[(t, True)]:>14d}{cross[(t, False)]:>18d}")
        n_pos = cross[("positive", True)] + cross[("positive", False)]
        say(f"\n   ** 정답키 의심 {len(suspect)}건 ** "
            f"(교사 positive {n_pos}건 중 {pct(len(suspect), n_pos)})")
        for s in sorted(suspect, key=lambda x: x["확신"] != "높음")[:10]:
            say(f"     {s['id'].split(':')[-1]:<22} 확신 {s['확신']:<4} {s['근거'][:52]}")
        say()

    # ── Q2. 라벨이 같은 건과 다른 건을 갈라서 ────────────────────────────────
    if q2_ans:
        say("── Q2  서술이 같은 내용인가  ★ 라벨로 갈라 본다")
        table = []
        for run in sorted({v.get("run", "") for v in key.values()}):
            for same_label, tag in ((True, "라벨 같음"), (False, "라벨 다름")):
                c = collections.Counter()
                for qid, a in q2_ans.items():
                    k = key.get(qid)
                    if not k or k.get("run") != run or k["라벨같음"] != same_label:
                        continue
                    if "dup" in qid:
                        continue
                    c[a.get("같은내용", "?")] += 1
                n = sum(c.values())
                if not n:
                    continue
                table.append((run, tag, n, c["같다"], c["일부"], c["다르다"]))
        say(f"   {'실행':22s}{'':10s}{'건':>4s}{'같다':>10s}{'일부':>8s}{'다르다':>8s}")
        for run, tag, n, s, p, d in table:
            say(f"   {run:22s}{tag:10s}{n:4d}{s:5d}({pct(s, n):>6s}){p:8d}{d:8d}")
        say()
        say("   ** 두 줄의 「같다」가 비슷하면 라벨일치가 서술을 예측하지 못한다는 뜻이다 **")
        say()

    # ── HTML ────────────────────────────────────────────────────────────────
    if args.out:
        command = "python -m H_eval.read_judge " + " ".join(shlex.quote(a) for a in sys.argv[1:])
        rows = "".join(
            f"<tr><td>{html.escape(s['id'].split(':')[-1])}</td>"
            f"<td>{html.escape(s['확신'])}</td><td>{html.escape(s['근거'])}</td></tr>"
            for s in suspect)
        body = "\n".join(html.escape(x) for x in lines)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"""<meta charset="utf-8"><title>심판 집계</title>
<style>body{{font:14px/1.7 -apple-system,'Segoe UI',sans-serif;margin:2rem auto;max-width:1000px;
 padding:0 1rem;color:#111}} pre{{background:#f6f7f9;border:1px solid #dde;border-radius:6px;
 padding:1rem;overflow:auto;font-size:.84rem}} table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #dcdfe4;padding:.3rem .5rem;font-size:.85rem;text-align:left}}
 thead th{{background:#f2f4f7}} code{{background:#f6f7f9;padding:.05rem .3rem}}
 .note{{background:#fffbe9;border-left:3px solid #e2c86a;padding:.6rem .9rem;font-size:.88rem}}</style>
<h1>심판 집계</h1>
<div class="note"><b>이것은 판정이 아니라 후보 목록이다.</b>
 <code>verdict</code>도 <code>sweep*.md</code>도 건드리지 않는다.
 아래 「정답키 의심」은 <b>사람이 조문을 보고 확인해야</b> 확정된다.<br>
 만든 명령 · <code>{html.escape(command)}</code></div>
<pre>{body}</pre>
<h2>정답키 의심 — 전체 {len(suspect)}건</h2>
<table><thead><tr><th>건</th><th>심판 확신</th><th>근거</th></tr></thead><tbody>{rows}</tbody></table>
""", encoding="utf-8")
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
