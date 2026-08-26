"""대상별로 가른 방향 혼동표. **GPU 안 쓴다.**

`runs/<이름>/<채점폴더>/records.jsonl`만 읽는다.

**왜 필요한가.** 지금 혼동표는 대상 7x7과 방향 5x5를 **따로** 그린다. 그래서
「다른 값 ↔ 늘었다 223건」이 나와도 그것이 **어느 대상 안에서 벌어진 일인지** 알 수 없다.
「기한·시점」의 30일→60일을 `늘었다`로 볼지 `다른 값`으로 볼지가 갈리는 문제는
**특정 대상 안에서의 방향 혼동**이므로, 그 교차 칸이 있어야 크기를 잰다.

**맞춰 보는 규칙(이 파일이 정하는 것이다).** 교사 라벨 한 항목이 한 줄이다.

    1. 모델이 **같은 `대상`**을 낸 항목이 있나 -> 없으면 `대상 못 맞힘`
    2. 있으면 그 항목들의 `방향`과 교사 `방향`을 견준다
    3. 같은 `대상`을 여럿 냈으면, 교사 방향이 그 안에 있으면 맞힌 것으로 본다.
       없으면 첫 항목의 방향을 적는다 (그런 건수는 머리말에 적는다)

`근거`는 자유 문장이라 안 본다 -- `sft.scoring.label_match`와 같다.

**세는 단위가 둘이고 뜻이 다르다.** 홀드아웃은 여러 실험이 **같은 건을 되풀이해 푼** 것이라

    측정  = 실험 수 x 건수. 실험들이 얼마나 자주 틀리나
    고유  = 서로 다른 (건, 라벨) 자리. **근거의 실제 폭이다**

**고유가 작으면 측정이 아무리 커도 근거는 그만큼이다.** 둘 다 적는다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):

    python -m H_eval.cross_confusion --runs 'runs/*-run2A' \
        --out H_eval/20260825__label_cross/교차혼동.html
"""
from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import shlex
import sys
from pathlib import Path

from sft.scoring import DIRECTIONS, TARGETS, label_pairs

# 맞춰 보기가 실패한 자리. 방향 5종 뒤에 이 세 칸이 더 붙는다.
MISS_TARGET = "대상 못 맞힘"
MISS_EMPTY = "라벨 없음"
MISS_BROKEN = "출력 깨짐"
EXTRA = [MISS_TARGET, MISS_EMPTY, MISS_BROKEN]


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def align(record: dict) -> list[tuple[str, str, str]]:
    """교사 라벨 한 항목마다 `(대상, 교사 방향, 모델이 낸 방향)`을 낸다.

    세 번째 자리에는 방향 5종이나 `EXTRA`의 셋 중 하나가 들어간다.
    """
    theirs = label_pairs(record.get("teacher_labels")) or []
    if not theirs:
        return []
    mine = label_pairs(record.get("labels"))
    if mine is None:                       # 파싱이 깨져 labels 자체가 없다
        return [(t, d, MISS_BROKEN) for t, d in theirs]
    if not mine:                           # 형식은 멀쩡한데 라벨을 안 냈다
        return [(t, d, MISS_EMPTY) for t, d in theirs]

    by_target = collections.defaultdict(list)
    for t, d in mine:
        by_target[t].append(d)

    out = []
    for t, d in theirs:
        got = by_target.get(t)
        if not got:
            out.append((t, d, MISS_TARGET))
        elif d in got:
            out.append((t, d, d))          # 교사 방향이 그 안에 있으면 맞힌 것
        else:
            out.append((t, d, got[0]))
    return out


def tally(run_dirs: list[Path], eval_dir: str) -> dict:
    cells = collections.Counter()          # (대상, 교사 방향, 모델 방향) -> 측정
    uniq = collections.defaultdict(set)    # 같은 키 -> {(실행 밖에서 같은 자리)}
    multi = 0                              # 같은 대상을 여럿 낸 건수
    used, items, broken = [], set(), 0

    for run in sorted(run_dirs):
        path = run / eval_dir / "records.jsonl"
        if not path.exists():
            continue
        used.append(run.name)
        for record in read_records(path):
            items.add(record["id"])
            if not isinstance(record.get("labels"), list):
                broken += 1
            mine = label_pairs(record.get("labels")) or []
            seen = collections.Counter(t for t, _ in mine)
            multi += sum(1 for t, n in seen.items() if n > 1)
            for i, (target, theirs, ours) in enumerate(align(record)):
                cells[(target, theirs, ours)] += 1
                uniq[(target, theirs, ours)].add((record["id"], i))

    return {"cells": cells, "uniq": {k: len(v) for k, v in uniq.items()},
            "runs": used, "items": len(items), "broken": broken, "multi": multi}


def esc(text) -> str:
    return html.escape(str(text))


def table_for(target: str, data: dict) -> str:
    """대상 하나의 5 x (5+3) 표. 행이 교사 방향, 열이 모델이 낸 방향이다."""
    cells, uniq = data["cells"], data["uniq"]
    cols = DIRECTIONS + EXTRA
    rows = [d for d in DIRECTIONS
            if any(cells.get((target, d, c)) for c in cols)]
    if not rows:
        return ""

    total = sum(v for (t, _, _), v in cells.items() if t == target)
    total_u = sum(v for (t, _, _), v in uniq.items() if t == target)
    hit = sum(cells.get((target, d, d), 0) for d in DIRECTIONS)

    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for d in rows:
        line = [f"<th class='r'>{esc(d)}</th>"]
        for c in cols:
            n = cells.get((target, d, c), 0)
            if not n:
                line.append("<td class='z'>·</td>")
                continue
            u = uniq.get((target, d, c), 0)
            klass = "ok" if c == d else ("mx" if c in DIRECTIONS else "no")
            line.append(f"<td class='{klass}'>{n}<span class='u'>{u}</span></td>")
        body.append(f"<tr>{''.join(line)}</tr>")

    rate = f"{hit / total:.1%}" if total else "—"
    return f"""<section>
<h3>{esc(target)} <small>방향 맞힘 {rate} · 측정 {total} · 고유 {total_u}</small></h3>
<table><thead><tr><th class='c'>교사 \\ 모델</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table></section>"""


def render(data: dict, args, command: str) -> str:
    cells, uniq = data["cells"], data["uniq"]
    order = sorted(TARGETS, key=lambda t: -sum(
        v for (x, _, _), v in cells.items() if x == t))
    tables = "".join(table_for(t, data) for t in order)

    total = sum(cells.values())
    total_u = sum(uniq.values())
    runs = ", ".join(data["runs"])

    # 방향만 틀린 자리를 큰 것부터. 이것이 이 판이 가리려는 것이다.
    swaps = sorted(((v, uniq[k], k) for k, v in cells.items()
                    if k[1] != k[2] and k[2] in DIRECTIONS), reverse=True)[:12]
    swap_rows = "".join(
        f"<tr><td>{esc(t)}</td><td>{esc(a)}</td><td>{esc(b)}</td>"
        f"<td class='n'>{n}</td><td class='n'>{u}</td></tr>"
        for n, u, (t, a, b) in swaps)

    return f"""<meta charset="utf-8"><title>대상별 방향 혼동표</title>
<style>
 body{{font:14px/1.7 -apple-system,'Segoe UI',sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#111}}
 h1{{font-size:1.4rem;margin:0 0 .3rem}} h3{{font-size:1rem;margin:1.6rem 0 .4rem}}
 small{{font-weight:400;color:#666}}
 .head{{background:#f6f7f9;border:1px solid #dde;border-radius:6px;padding:.9rem 1.1rem;margin:1rem 0 1.6rem}}
 .head dt{{float:left;width:8.5rem;clear:left;color:#555;font-size:.86rem}}
 .head dd{{margin:0 0 .35rem 8.5rem;font-size:.86rem}}
 .head code{{background:#fff;border:1px solid #ddd;border-radius:3px;padding:.05rem .3rem;font-size:.82rem}}
 table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
 th,td{{border:1px solid #dcdfe4;padding:.3rem .5rem;text-align:right;font-size:.85rem}}
 thead th{{background:#f2f4f7;font-weight:600;text-align:center}}
 th.r{{text-align:left;background:#fafbfc;font-weight:600}} th.c{{text-align:left;background:#e8ebef}}
 td.ok{{background:#eef7ee}} td.mx{{background:#fdf3e3}} td.no{{background:#f7f7f8;color:#888}}
 td.z{{color:#ccc;text-align:center}} .u{{color:#888;font-size:.75rem;margin-left:.35rem}}
 .n{{text-align:right}}
 .note{{color:#555;font-size:.86rem;background:#fffbe9;border-left:3px solid #e2c86a;padding:.6rem .9rem;margin:1rem 0}}
</style>
<h1>대상별 방향 혼동표</h1>
<dl class="head">
 <dt>가리려는 것</dt><dd>한 대상 안에서 <b>방향</b>이 어디로 새는가.
   특히 <code>기한·시점</code>의 <code>늘었다</code> ↔ <code>다른 값</code></dd>
 <dt>잰 것</dt><dd>{esc(runs)}</dd>
 <dt>채점 폴더</dt><dd><code>{esc(args.eval_dir)}</code> · 홀드아웃 {data['items']}건 · 실행 {len(data['runs'])}개</dd>
 <dt>분모</dt><dd><b>교사 라벨 항목</b>이 한 줄이다 (건이 아니다).
   측정 {total} · 고유 {total_u}.
   교사가 라벨을 안 단 건은 애초에 안 들어온다</dd>
 <dt>깨진 출력</dt><dd>{data['broken']}건은 <code>{MISS_BROKEN}</code> 칸으로 <b>뺀 것이 아니라 넣었다</b>.
   빼면 형식이 깨진 실험일수록 방향 정확도가 좋아 보인다</dd>
 <dt>같은 대상 중복</dt><dd>{data['multi']}건. 첫 항목의 방향을 적었다</dd>
 <dt>만든 날 · 명령</dt><dd>{esc(args.date)} · <code>{esc(command)}</code></dd>
</dl>

<div class="note"><b>큰 숫자가 측정, 옆의 작은 숫자가 고유다.</b>
 홀드아웃 {data['items']}건을 실험 {len(data['runs'])}개가 되풀이해 풀었으므로
 <b>측정은 같은 자리를 여러 번 센 값이다.</b> 근거의 폭은 고유 쪽이다.</div>

<h3>방향만 틀린 자리 — 큰 것부터</h3>
<table><thead><tr><th class='c'>대상</th><th>교사</th><th>모델</th><th>측정</th><th>고유</th></tr></thead>
<tbody>{swap_rows}</tbody></table>

{tables}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="대상별로 가른 방향 혼동표")
    ap.add_argument("--runs", action="append", metavar="GLOB",
                    help="실행 폴더 glob. 여러 번 줄 수 있다 (기본 runs/*-run2A)")
    ap.add_argument("--eval-dir", default="eval-mof-motie",
                    help="채점 폴더 이름. 자가 여기서 갈린다")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default="", help="머리말에 적을 날짜")
    args = ap.parse_args()

    patterns = args.runs or ["runs/*-run2A"]
    run_dirs = [Path(p) for pat in patterns for p in glob.glob(pat)]
    if not run_dirs:
        sys.exit(f"실행 폴더를 못 찾았다: {patterns}")

    data = tally(run_dirs, args.eval_dir)
    if not data["runs"]:
        sys.exit(f"'{args.eval_dir}/records.jsonl'이 있는 폴더가 없다")

    command = "python -m H_eval.cross_confusion " + " ".join(
        shlex.quote(a) for a in sys.argv[1:])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data, args, command), encoding="utf-8")

    print(f"저장: {args.out}")
    print(f"  실행 {len(data['runs'])}개 · 홀드아웃 {data['items']}건 · "
          f"교사 라벨 측정 {sum(data['cells'].values())} · "
          f"고유 {sum(data['uniq'].values())}")


if __name__ == "__main__":
    main()
