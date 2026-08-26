"""라벨이 **방향만** 어긋난 자리를 사람이 읽고 누가 맞는지 정하는 판. **GPU 안 쓴다.**

**무엇을 정하려는 판인가.** 「등가표를 만들 것인가」다. 채점기가 `(대상, 방향)` 쌍을
정확일치로 보는데, 경계가 흐린 쌍이 있으면 억울하게 틀린 것으로 잡힌다. 그런데
**어긋남에는 뜻이 다른 셋이 섞여 있다.**

    교사가 맞다        -> 모델이 틀린 것이다. 자는 그대로 둔다
    모델이 맞다        -> 교사 라벨이 틀린 것이다. `data_collect` 소관이다
    둘 다 말이 된다    -> 경계가 흐린 것이다. **여기가 등가표의 근거다**

**셋을 코드로는 못 가른다.** 조문을 읽어야 알 수 있어서 사람이 판다.

**기본은 「적용 범위」의 `늘었다` ↔ `줄었다`다.** 이건 경계가 흐린 것이 아니라
**정반대로 읽은 것**이라, 등가표로 접으면 안 되는 쌍이다. 그래서 여기서 갈라야 할 것은
「접을까」가 아니라 **「누가 틀렸나」**다. 적용 대상이 넓어진 것과 좁아진 것은 실무
영향이 뒤집히므로, 교사가 틀렸다면 그게 제일 큰 문제다.

**같은 자리를 여러 실험이 되풀이해 푼다.** 그래서 카드마다 **몇 개가 그렇게 답했는지**를
같이 적는다 -- 14개 중 14개가 교사와 다르게 답했으면 **교사 쪽을 의심할 근거**가 된다.

**눈가림은 완전하지 않다.** 좌우를 섞고 `근거`의 `이전:`·`이후:` 말머리를 떼지만,
문투로 눈치챌 수 있다. **그럴 때는 알고 보되 판단은 조문에서 한다.**

사용 (**저장소 뿌리에서 `-m`으로 부른다**):

    python -m H_eval.label_review --out H_eval/20260825__label_cross/판정판.html
    python -m H_eval.label_review --cell '절차·요건:다른 값:새로 생겼다' --out ...
"""
from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import random
import re
import shlex
import sys
from pathlib import Path

from cli.readout import diff_html
from sft.scoring import DIRECTIONS, label_pairs

HOLDOUT = "data/20260821__annotate__v2.2-run2A/holdout.jsonl"
# 기본으로 볼 칸. 「적용 범위」의 뒤집힘 한 쌍이다. `--cell`로 더 넣는다.
DEFAULT_CELLS = ["적용 범위:늘었다:줄었다", "적용 범위:줄었다:늘었다"]

_LEAD = re.compile(r"^\s*이전\s*:\s*|\s*→\s*이후\s*:\s*")


def strip_tell(text: str) -> str:
    """`근거`에서 교사 쪽 말머리를 뗀다. 눈가림을 조금이라도 지키려는 것이다."""
    return _LEAD.sub(" → ", str(text or "")).strip(" →")


def collect(run_dirs: list[Path], eval_dir: str, cells: set[tuple[str, str, str]]) -> dict:
    """칸에 걸리는 자리를 모은다. 열쇠는 `(건 id, 교사 라벨 순번)`이다."""
    slots: dict[tuple[str, int], dict] = {}
    runs = []
    for run in sorted(run_dirs):
        path = run / eval_dir / "records.jsonl"
        if not path.exists():
            continue
        runs.append(run.name)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                theirs = label_pairs(rec.get("teacher_labels")) or []
                mine = label_pairs(rec.get("labels"))
                if not theirs or not mine:
                    continue
                by_target = collections.defaultdict(list)
                for j, (t, d) in enumerate(mine):
                    by_target[t].append((j, d))
                for i, (t, d) in enumerate(theirs):
                    got = by_target.get(t)
                    if not got or d in [x[1] for x in got]:
                        continue          # 못 맞힌 것이 아니거나 맞힌 것이다
                    j, ours = got[0]
                    if (t, d, ours) not in cells:
                        continue
                    slot = slots.setdefault((rec["id"], i), {
                        "target": t, "teacher_dir": d, "teacher_label": rec["teacher_labels"][i],
                        "votes": collections.Counter(), "by_dir": {}, "seen": 0})
                    slot["votes"][ours] += 1
                    slot["by_dir"].setdefault(ours, {
                        "run": run.name, "label": rec["labels"][j],
                        "direct_impact": rec.get("direct_impact", "")})
    return {"slots": slots, "runs": runs}


def card(no: int, key, slot: dict, holdout: dict, rnd: random.Random, total_runs: int) -> str:
    row = holdout.get(key[0], {})
    left_html, right_html = diff_html(row.get("before", ""), row.get("after", ""))

    top_dir, votes = slot["votes"].most_common(1)[0]
    model = slot["by_dir"][top_dir]

    teacher_side = {"방향": slot["teacher_dir"],
                    "근거": strip_tell(slot["teacher_label"].get("근거", "")),
                    "문장": row.get("direct_impact", "")}
    model_side = {"방향": top_dir,
                  "근거": strip_tell(model["label"].get("근거", "")),
                  "문장": model["direct_impact"]}

    # 좌우를 섞는다. 어느 쪽이 교사인지는 답을 저장할 때 같이 나간다.
    flip = rnd.random() < 0.5
    a, b = (model_side, teacher_side) if flip else (teacher_side, model_side)
    a_is = "model" if flip else "teacher"

    def block(tag: str, side: dict) -> str:
        return f"""<div class="side">
  <div class="tag">{tag}</div>
  <div class="dir">{html.escape(side['방향'])}</div>
  <div class="lab">근거</div><div class="txt">{html.escape(side['근거']) or '—'}</div>
  <div class="lab">direct_impact</div><div class="txt">{html.escape(side['문장']) or '—'}</div>
</div>"""

    return f"""<article class="card" data-i="{no}" data-id="{html.escape(key[0])}"
  data-slot="{key[1]}" data-a="{a_is}" data-target="{html.escape(slot['target'])}"
  data-teacher="{html.escape(slot['teacher_dir'])}" data-model="{html.escape(top_dir)}"
  data-votes="{votes}" data-runs="{total_runs}">
 <h3>{no}. {html.escape(slot['target'])}
   <small>{html.escape(key[0])} · 라벨 {key[1] + 1}번째 ·
   <b>{votes}/{total_runs}</b>개 실험이 같은 답을 냈다</small></h3>
 <div class="diff"><div><div class="lab">개정 전</div>{left_html}</div>
   <div><div class="lab">개정 후</div>{right_html}</div></div>
 <div class="sides">{block('갑', a)}{block('을', b)}</div>
 <div class="ask">
   <button data-v="갑">갑이 맞다</button>
   <button data-v="을">을이 맞다</button>
   <button data-v="둘다">둘 다 말이 된다</button>
   <button data-v="둘다틀림">둘 다 틀렸다</button>
   <input class="memo" placeholder="메모 (왜 그렇게 봤나)">
 </div>
</article>"""


def render(cards: list[str], data: dict, args, command: str, cells: list[str]) -> str:
    return f"""<meta charset="utf-8"><title>라벨 방향 판정판</title>
<style>
 body{{font:14px/1.7 -apple-system,'Segoe UI',sans-serif;margin:2rem auto;max-width:1000px;padding:0 1rem;color:#111}}
 h1{{font-size:1.4rem;margin:0 0 .3rem}} h3{{font-size:.95rem;margin:0 0 .7rem}}
 small{{font-weight:400;color:#666}}
 .head{{background:#f6f7f9;border:1px solid #dde;border-radius:6px;padding:.9rem 1.1rem;margin:1rem 0}}
 .head dt{{float:left;width:8.5rem;clear:left;color:#555;font-size:.86rem}}
 .head dd{{margin:0 0 .35rem 8.5rem;font-size:.86rem}}
 code{{background:#fff;border:1px solid #ddd;border-radius:3px;padding:.05rem .3rem;font-size:.82rem}}
 .rule{{background:#fffbe9;border-left:3px solid #e2c86a;padding:.6rem .9rem;margin:1rem 0;font-size:.88rem}}
 .card{{border:1px solid #dcdfe4;border-radius:8px;padding:1rem 1.2rem;margin:1.4rem 0}}
 .card.done{{background:#f6faf6;border-color:#bcd8bc}}
 .diff{{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:.9rem}}
 .diff>div{{background:#fafbfc;border:1px solid #e6e8ec;border-radius:5px;padding:.6rem .8rem;
   font-size:.84rem;max-height:15rem;overflow:auto;white-space:pre-wrap;word-break:break-all}}
 del{{background:#ffe1e1;text-decoration:line-through}} ins{{background:#dcf3dc;text-decoration:none}}
 .lab{{color:#777;font-size:.75rem;letter-spacing:.03em;margin:.5rem 0 .15rem}}
 .sides{{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}}
 .side{{border:1px solid #e0e3e8;border-radius:5px;padding:.6rem .8rem;background:#fff}}
 .tag{{font-weight:700;font-size:1rem}} .dir{{font-size:1.05rem;font-weight:600;color:#2a4d8f}}
 .txt{{font-size:.84rem;white-space:pre-wrap}}
 .ask{{margin-top:.9rem;display:flex;gap:.4rem;flex-wrap:wrap;align-items:center}}
 .ask button{{padding:.35rem .8rem;font-size:.85rem;border:1px solid #c8ccd3;border-radius:5px;
   background:#fff;cursor:pointer}}
 .ask button.on{{background:#2a4d8f;color:#fff;border-color:#2a4d8f}}
 .memo{{flex:1;min-width:14rem;padding:.35rem .5rem;font-size:.85rem;border:1px solid #d5d8dd;border-radius:5px}}
 #bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde;padding:.6rem 0;z-index:9}}
 #bar button{{padding:.3rem .7rem;font-size:.85rem;margin-right:.3rem}}
 textarea{{width:100%;height:8rem;font:12px/1.5 ui-monospace,monospace;margin-top:.5rem}}
</style>
<h1>라벨 방향 판정판</h1>
<dl class="head">
 <dt>가리려는 것</dt><dd>방향이 어긋난 자리에서 <b>누가 맞나</b>.
   교사냐 · 모델이냐 · 둘 다 말이 되냐</dd>
 <dt>본 칸</dt><dd>{" · ".join(f"<code>{html.escape(c)}</code>" for c in cells)}</dd>
 <dt>잰 것</dt><dd>{html.escape(", ".join(data["runs"]))}</dd>
 <dt>채점 폴더</dt><dd><code>{html.escape(args.eval_dir)}</code> · 홀드아웃 <code>{html.escape(args.holdout)}</code></dd>
 <dt>분모</dt><dd><b>고유 {len(cards)}자리.</b> 열쇠는 (건 id, 교사 라벨 순번)이다.
   같은 자리를 여러 실험이 푼 것은 한 장으로 묶고, 몇 개가 같은 답을 냈는지만 적었다.
   모델 쪽은 <b>제일 많이 나온 방향</b> 하나를 대표로 세웠다</dd>
 <dt>눈가림</dt><dd>갑·을 자리를 건마다 섞었고 <code>근거</code>의 <code>이전:</code>·
   <code>이후:</code> 말머리를 뗐다. <b>문투로 새어나갈 수 있다</b> — 그럴 때는 알고 보되
   판단은 조문에서 한다</dd>
 <dt>만든 날 · 명령</dt><dd>{html.escape(args.date)} · <code>{html.escape(command)}</code></dd>
</dl>

<div class="rule"><b>답이 뜻하는 것.</b>
 <b>교사가 맞다</b> → 모델 잘못이다. 자는 그대로 둔다 ·
 <b>모델이 맞다</b> → <b>교사 라벨이 틀렸다.</b> <code>data_collect</code>로 넘긴다 ·
 <b>둘 다 말이 된다</b> → 경계가 흐리다. <b>여기가 등가표의 근거다</b> ·
 <b>둘 다 틀렸다</b> → 조문을 아무도 못 읽었다<br>
 <b>「적용 범위」의 늘었다 ↔ 줄었다는 정반대로 읽은 것이라 등가표로 접으면 안 된다.</b>
 여기서 「둘 다 말이 된다」가 많이 나오면 그건 등가표가 아니라
 <b>규칙서에서 그 칸의 뜻이 흐리다</b>는 뜻이다.</div>

<div id="bar">
 <b id="cnt">0/{len(cards)}</b>
 <button id="tsv">TSV</button><button id="json">JSON</button>
 <button id="save">파일로 저장</button><button id="reset">초기화</button>
 <span style="color:#a33;font-size:.82rem">끝나면 반드시 「파일로 저장」</span>
 <textarea id="out" readonly></textarea>
</div>

{"".join(cards)}

<script>
const KEY = "H_eval_label_review";
let ans = {{}};
try {{ ans = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ ans = {{}}; }}
let mode = "tsv";

function rows() {{
  return [...document.querySelectorAll(".card")].map(c => {{
    const a = ans[c.dataset.i] || {{}};
    // 갑·을은 화면 자리다. 저장할 때 교사/모델로 되돌린다.
    let who = a.v || "";
    if (who === "갑") who = c.dataset.a === "teacher" ? "교사" : "모델";
    else if (who === "을") who = c.dataset.a === "teacher" ? "모델" : "교사";
    return {{
      no: c.dataset.i, id: c.dataset.id, slot: c.dataset.slot,
      대상: c.dataset.target, 교사방향: c.dataset.teacher, 모델방향: c.dataset.model,
      같은답: c.dataset.votes + "/" + c.dataset.runs,
      판정: who, 메모: (a.m || "").replace(/\\t/g, " ")
    }};
  }});
}}
function dump() {{
  const r = rows();
  document.getElementById("out").value = mode === "json"
    ? JSON.stringify(r, null, 1)
    : [Object.keys(r[0] || {{}}).join("\\t"),
       ...r.map(x => Object.values(x).join("\\t"))].join("\\n");
  document.getElementById("cnt").textContent =
    r.filter(x => x.판정).length + "/" + r.length;
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(ans)); dump(); }}

document.querySelectorAll(".card").forEach(c => {{
  const i = c.dataset.i, a = ans[i] || {{}};
  c.querySelectorAll(".ask button").forEach(b => {{
    if (a.v === b.dataset.v) {{ b.classList.add("on"); c.classList.add("done"); }}
    b.onclick = () => {{
      c.querySelectorAll(".ask button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); c.classList.add("done");
      ans[i] = Object.assign({{}}, ans[i], {{v: b.dataset.v}}); save();
    }};
  }});
  const m = c.querySelector(".memo");
  m.value = a.m || "";
  // 글자마다 다시 그리면 포커스를 잃는다. 저장만 하고 화면은 그대로 둔다.
  m.oninput = () => {{ ans[i] = Object.assign({{}}, ans[i], {{m: m.value}});
                      localStorage.setItem(KEY, JSON.stringify(ans)); }};
}});
document.getElementById("tsv").onclick = () => {{ mode = "tsv"; dump(); }};
document.getElementById("json").onclick = () => {{ mode = "json"; dump(); }};
document.getElementById("reset").onclick = () => {{
  if (confirm("판정을 전부 지운다")) {{ ans = {{}}; localStorage.removeItem(KEY); location.reload(); }}
}};
// 브라우저 저장소만으로는 부족하다. 저장소를 비우거나 다른 기기에서 열면 사라진다.
document.getElementById("save").onclick = () => {{
  const blob = new Blob([document.getElementById("out").value],
                        {{type: "text/plain;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "답." + (mode === "json" ? "json" : "tsv");
  a.click();
}};
dump();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="라벨 방향 판정판")
    ap.add_argument("--runs", action="append", metavar="GLOB",
                    help="실행 폴더 glob (기본 runs/*-run2A)")
    ap.add_argument("--eval-dir", default="eval-mof-motie")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--cell", action="append", metavar="대상:교사방향:모델방향",
                    help=f"볼 칸. 여러 번 줄 수 있다 (기본 {' , '.join(DEFAULT_CELLS)})")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--seed", type=int, default=42, help="갑·을 자리를 섞는 씨앗")
    args = ap.parse_args()

    raw_cells = args.cell or DEFAULT_CELLS
    cells = set()
    for spec in raw_cells:
        parts = spec.split(":")
        if len(parts) != 3 or parts[1] not in DIRECTIONS or parts[2] not in DIRECTIONS:
            sys.exit(f"칸 모양이 틀렸다: {spec}  (대상:교사방향:모델방향)")
        cells.add(tuple(parts))

    run_dirs = [Path(p) for pat in (args.runs or ["runs/*-run2A"]) for p in glob.glob(pat)]
    data = collect(run_dirs, args.eval_dir, cells)
    if not data["runs"]:
        sys.exit(f"'{args.eval_dir}/records.jsonl'이 있는 폴더가 없다")
    if not data["slots"]:
        sys.exit("그 칸에 걸리는 자리가 없다. --cell을 다시 본다")

    holdout = {}
    with (Path(args.holdout)).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            holdout[row["id"]] = row

    rnd = random.Random(args.seed)
    # 여러 실험이 같은 답을 낸 것부터. 표가 셀수록 교사 쪽을 의심할 근거가 크다.
    keys = sorted(data["slots"], key=lambda k: -sum(data["slots"][k]["votes"].values()))
    cards = [card(i + 1, k, data["slots"][k], holdout, rnd, len(data["runs"]))
             for i, k in enumerate(keys)]

    command = "python -m H_eval.label_review " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(cards, data, args, command, raw_cells), encoding="utf-8")

    print(f"저장: {args.out}")
    print(f"  실행 {len(data['runs'])}개 · 볼 자리 {len(cards)}개 (고유)")


if __name__ == "__main__":
    main()
