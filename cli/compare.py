"""두 실험의 `direct_impact` 문장을 **눈가림으로 맞붙여 사람이 채점하는 판**을 만든다.

**이 저장소의 채점기는 문장을 못 잰다.** AM 다섯은 형식 검사, `교사일치`는 `judgement`
한 칸, `라벨일치`는 `(대상, 방향)` 집합이다. 셋 다 산출물인 문장 자체는 안 본다.
그리고 **라벨이 달라도 문장은 얼추 같은 경우가 흔하므로**, 라벨일치를 문장 품질의
대리값으로 쓰면 실제보다 낮게 본다. 그래서 여기서는 사람이 직접 읽고 고른다.

**한 카드에 두 실험의 문장을 나란히 놓는다.** 한 실험씩 따로 채점하면 같은 조문을 두 번
읽어야 하고, 앞에서 본 문장이 뒤 판단에 남는다. 나란히 놓으면 한 번 읽고 한 번 고른다.

**어느 쪽이 어느 실험인지 가린다.** `A`·`B` 자리는 `id`로 정해지므로 다시 만들어도 같고,
사람이 "2차니까 낫겠지"로 기울지 않는다. 가림을 풀면 헤더의 눈금이 바뀐다.

**교사 해석은 접어 둔다.** 먼저 조문을 보고 스스로 정한 뒤 펴야, 교사 문장을 베낀 쪽으로
점수가 쏠리지 않는다. 교사는 정답이 아니라 교사 모델의 출력이다.

점수는 브라우저에 남고(`localStorage`) 버튼으로 TSV·JSON을 파일로 떨어뜨린다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.compare --a runs/delora --b runs/delora-r2 \\
        --label '1차 562건' '2차 1,655건' --out runs/문장채점.html
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from cli.readout import diff_html, label_set, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = "data/20260811__annotate__v2.2/holdout.jsonl"


def side(rid: str) -> bool:
    """이 건에서 첫째 실험을 `A`에 놓을지. `id`로 정해지므로 **다시 만들어도 같다.**

    난수를 쓰면 판을 다시 만들 때마다 자리가 바뀌어, 채워 둔 점수가 딴 문장에 붙는다.
    """
    return int(hashlib.md5(rid.encode()).hexdigest(), 16) % 2 == 0


def sentence(row: dict) -> str:
    return (row.get("direct_impact") or "").strip()


def build(a_dir: Path, b_dir: Path, labels: tuple[str, str], holdout: Path) -> tuple[str, dict]:
    teacher = read_jsonl(holdout)
    a_rows = read_jsonl(a_dir / "eval" / "records.jsonl")
    b_rows = read_jsonl(b_dir / "eval" / "records.jsonl")

    items, both_empty = [], 0
    for n, rid in enumerate(i for i in teacher if i in a_rows and i in b_rows):
        t, first, second = teacher[rid], a_rows[rid], b_rows[rid]
        # 둘 다 문장을 안 냈으면 견줄 것이 없다. 세어서 알려주되 판에는 올린다 --
        # 빼 버리면 분모가 조용히 달라져 "몇 건 중 몇 건"을 못 쓴다.
        if not sentence(first) and not sentence(second):
            both_empty += 1

        left, right = (first, second) if side(rid) else (second, first)
        which = ("a", "b") if side(rid) else ("b", "a")
        before, after = diff_html(t.get("before", ""), t.get("after", ""))
        items.append({
            "n": n, "id": rid,
            "short": t.get("before_id", rid).split("-")[-1],
            "beforeHtml": before, "afterHtml": after,
            "teacher": {
                "judgement": t.get("judgement", ""),
                "labels": [f"{x} · {y}" for x, y in sorted(label_set(t.get("labels")))],
                "sentence": sentence(t),
            },
            "A": {"judgement": left.get("judgement", ""), "sentence": sentence(left),
                  "labelSame": label_set(left.get("labels")) == label_set(t.get("labels")),
                  "labels": [f"{x} · {y}" for x, y in sorted(label_set(left.get("labels")))]},
            "B": {"judgement": right.get("judgement", ""), "sentence": sentence(right),
                  "labelSame": label_set(right.get("labels")) == label_set(t.get("labels")),
                  "labels": [f"{x} · {y}" for x, y in sorted(label_set(right.get("labels")))]},
            # 어느 자리가 어느 실험인지. 가림을 풀 때와 내보낼 때만 쓴다
            "key": {"A": which[0], "B": which[1]},
        })

    data = {"a": {"run": a_dir.name, "label": labels[0]},
            "b": {"run": b_dir.name, "label": labels[1]},
            "items": items}
    stats = {"총": len(items), "둘다빈문장": both_empty}
    return page(data, stats), stats


def page(data: dict, stats: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>문장 채점 · {html.escape(data['a']['run'])} vs {html.escape(data['b']['run'])}</title>
<style>
  :root{{
    --bg:#f6f7f9; --card:#fff; --ink:#16191d; --muted:#666e7a; --line:#dfe3e8;
    --del:#fdecec; --delink:#a02020; --ins:#e8f6ec; --insink:#12692e;
    --warn:#fff6e0; --warnline:#e0b34d; --accent:#2a5bd7; --a:#7b4fd1; --b:#0f7b6c;
  }}
  @media (prefers-color-scheme:dark){{
    :root{{
      --bg:#14161a; --card:#1c1f24; --ink:#e6e8ec; --muted:#98a0ab; --line:#2c3038;
      --del:#3b1f1f; --delink:#ff9a90; --ins:#17301f; --insink:#7fd39a;
      --warn:#332a12; --warnline:#8a6f2c; --accent:#7aa2f7; --a:#b18cf0; --b:#5fc8b6;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.65 -apple-system,"Segoe UI","Malgun Gothic",sans-serif}}
  header{{position:sticky;top:0;z-index:10;background:var(--card);
         border-bottom:1px solid var(--line);padding:12px 20px;
         display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
  h1{{font-size:15px;margin:0;font-weight:700}}
  .meta{{color:var(--muted);font-size:12.5px}}
  .prog{{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:700}}
  button{{font:inherit;padding:5px 12px;border:1px solid var(--line);background:var(--card);
         color:var(--ink);border-radius:6px;cursor:pointer}}
  button:hover{{border-color:var(--accent);color:var(--accent)}}
  main{{max-width:1080px;margin:0 auto;padding:20px}}
  .lede{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:14px 18px;margin-bottom:16px;color:var(--muted);font-size:13.5px}}
  .lede b{{color:var(--ink)}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:16px 18px;margin-bottom:14px}}
  .card.done{{border-left:4px solid var(--insink)}}
  .card.cur{{box-shadow:0 0 0 2px var(--accent)}}
  .row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}}
  .tag{{font-size:12px;padding:2px 9px;border-radius:20px;background:var(--bg);color:var(--muted)}}
  .tag.same{{background:var(--ins);color:var(--insink)}}
  .tag.diff{{background:var(--del);color:var(--delink)}}
  .num{{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums}}
  .lbl{{font-size:12px;color:var(--muted);margin:12px 0 4px;letter-spacing:.04em}}
  .txt{{font:14px/1.85 ui-monospace,"D2Coding",Consolas,monospace;
       white-space:pre-wrap;word-break:break-word;background:var(--bg);
       border:1px solid var(--line);border-radius:7px;padding:10px 12px}}
  del{{background:var(--del);color:var(--delink);text-decoration:line-through}}
  ins{{background:var(--ins);color:var(--insink);text-decoration:none}}
  .sent{{border:1px solid var(--line);border-radius:7px;padding:11px 13px;margin-top:6px;
        display:flex;gap:12px;align-items:baseline}}
  .sent .who{{flex:0 0 22px;font-weight:700;font-size:13px}}
  .sent.A{{border-left:3px solid var(--a)}} .sent.A .who{{color:var(--a)}}
  .sent.B{{border-left:3px solid var(--b)}} .sent.B .who{{color:var(--b)}}
  .sent p{{margin:0;flex:1}}
  .none{{color:var(--muted);font-style:italic}}
  details{{margin-top:12px;border-top:1px solid var(--line);padding-top:8px}}
  summary{{cursor:pointer;color:var(--muted);font-size:12.5px;user-select:none}}
  summary:hover{{color:var(--ink)}}
  .judge{{background:var(--bg);border-radius:7px;padding:10px 12px;margin-top:8px;font-size:14px}}
  .score{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px;
         border-top:1px solid var(--line);padding-top:12px}}
  .g{{display:flex;gap:6px}}
  .score button.on{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  select,input[type=text]{{font:inherit;font-size:13px;padding:5px 9px;border-radius:6px;
    border:1px solid var(--line);background:var(--card);color:var(--ink)}}
  input.memo{{flex:1;min-width:140px}}
  textarea{{width:100%;height:150px;font:12.5px/1.6 ui-monospace,Consolas,monospace;
    border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--card);
    color:var(--ink);margin-top:10px}}
  body:not(.open) .reveal{{visibility:hidden}}
</style>
</head>
<body>
<header>
  <h1>문장 채점</h1>
  <span class="meta" id="meta"></span>
  <label class="meta"><input type="checkbox" id="open"> 가림 풀기</label>
  <button id="tsv">TSV</button><button id="json">JSON</button>
  <button id="download">파일로 저장</button><button id="reset">초기화</button>
  <span class="prog" id="prog"></span>
</header>
<main>
  <div class="lede">
    <b>같은 조문에 두 실험이 낸 문장을 나란히 놓았다.</b> 어느 쪽이 어느 실험인지는 가려져
    있다 — 자리는 건마다 다르다. <b>개정 전후를 먼저 보고 고른 뒤</b>, 필요하면 교사 해석을
    펴서 대조한다. 교사는 정답이 아니라 교사 모델의 출력이다.<br><br>
    <b>키보드</b> — <code>1</code> A가 낫다 · <code>2</code> B가 낫다 ·
    <code>3</code> 둘이 같은 뜻 · <code>4</code> 둘 다 틀렸다 · <code>j</code>/<code>k</code> 이동.
    고르면 다음 건으로 넘어간다. 점수는 브라우저에 남지만
    <b>끝나면 「파일로 저장」을 눌러 떨어뜨려 둘 것.</b>
  </div>
  <div id="list"></div>
  <textarea id="out" readonly></textarea>
</main>
<script>
const DATA = {blob};
const KEY = "sentence-compare-" + DATA.a.run + "-" + DATA.b.run;
let marks = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let cur = 0, mode = "tsv";

const CHOICE = {{1: "A", 2: "B", 3: "같음", 4: "둘다틀림"}};
// 왜 그렇게 봤는지. 다음 판에서 무엇을 고쳐야 하는지는 이 분포에서 나온다.
const WHY = ["", "한쪽이 사실을 지어냄", "한쪽이 방향을 반대로 읽음",
             "한쪽이 덜 구체적", "한쪽만 주체를 빠뜨림", "비문", "기타"];

document.getElementById("meta").textContent =
  "홀드아웃 " + DATA.items.length + "건 · " + DATA.a.label + " vs " + DATA.b.label;

function esc(s){{
  return (s == null ? "" : String(s)).replace(/[&<>]/g,
    c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
}}
function attr(s){{ return esc(s).replace(/"/g, "&quot;"); }}
function runOf(it, seat){{ return DATA[it.key[seat]].label; }}

function sentBox(it, seat){{
  const s = it[seat];
  const body = s.sentence ? esc(s.sentence) : "<span class=none>문장 없음</span>";
  return `<div class="sent ${{seat}}"><span class="who">${{seat}}</span>
    <p>${{body}}<br>
      <span class="num">판정 ${{esc(s.judgement) || "—"}}</span>
      <span class="tag ${{s.labelSame ? "same" : "diff"}}">라벨 ${{s.labelSame ? "같음" : "다름"}}</span>
      <span class="num reveal">${{esc(runOf(it, seat))}}</span>
    </p></div>`;
}}

function render(){{
  document.getElementById("list").innerHTML = DATA.items.map(it => {{
    const m = marks[it.id] || {{}};
    const tl = it.teacher.labels.map(x => `<span class="tag">${{esc(x)}}</span>`).join("") || "—";
    return `
    <div class="card ${{m.pick ? "done" : ""}} ${{it.n === cur ? "cur" : ""}}" id="c${{it.n}}">
      <div class="row">
        <span class="num">#${{it.n + 1}}</span>
        <span class="num">${{esc(it.short)}}</span>
        <span class="tag">교사 판정 ${{esc(it.teacher.judgement)}}</span>
      </div>
      <div class="lbl">개정 전</div><div class="txt">${{it.beforeHtml}}</div>
      <div class="lbl">개정 후</div><div class="txt">${{it.afterHtml}}</div>
      <div class="lbl">두 실험이 낸 문장</div>
      ${{sentBox(it, "A")}}
      ${{sentBox(it, "B")}}
      <details>
        <summary>교사 해석 보기 (고른 뒤에 펴는 것을 권함)</summary>
        <div class="judge">
          <div><b>labels</b> ${{tl}}</div>
          <div style="margin-top:8px"><b>direct_impact</b><br>${{
            esc(it.teacher.sentence) || "<span class=none>없음</span>"}}</div>
        </div>
      </details>
      <div class="score">
        <span class="g">
          ${{[1,2,3,4].map(k =>
            `<button data-id="${{attr(it.id)}}" data-v="${{CHOICE[k]}}"
               class="${{m.pick === CHOICE[k] ? "on" : ""}}">${{k}} ${{
               {{1:"A가 낫다",2:"B가 낫다",3:"둘이 같은 뜻",4:"둘 다 틀렸다"}}[k]}}</button>`
          ).join("")}}
        </span>
        <select data-id="${{attr(it.id)}}" class="why">
          ${{WHY.map(w => `<option value="${{attr(w)}}" ${{m.why === w ? "selected" : ""}}>${{
            esc(w) || "왜 그렇게 봤나 —"}}</option>`).join("")}}
        </select>
        <input type="text" class="memo" data-id="${{attr(it.id)}}" placeholder="메모"
               value="${{attr(m.memo || "")}}">
      </div>
    </div>`;
  }}).join("");

  const done = DATA.items.filter(it => (marks[it.id] || {{}}).pick);
  const count = p => done.filter(it => marks[it.id].pick === p).length;
  document.getElementById("prog").textContent =
    `채점 ${{done.length}} / ${{DATA.items.length}}` + (done.length
      ? `  ·  같은 뜻 ${{count("같음")}}  ·  A ${{count("A")}}  B ${{count("B")}}  ·  둘 다 틀림 ${{count("둘다틀림")}}`
      : "");
  dump();
}}

function save(){{ localStorage.setItem(KEY, JSON.stringify(marks)); }}
function set(id, patch){{
  marks[id] = Object.assign({{}}, marks[id], patch); save(); render();
}}
function goto(i){{
  cur = Math.max(0, Math.min(DATA.items.length - 1, i));
  document.getElementById("c" + cur)?.scrollIntoView({{block: "center", behavior: "smooth"}});
}}

document.addEventListener("click", e => {{
  const button = e.target.closest("button[data-id]");
  if (!button) return;
  const it = DATA.items.find(x => x.id === button.dataset.id);
  set(button.dataset.id, {{pick: button.dataset.v}});
  goto(it.n + 1);
}});
document.addEventListener("change", e => {{
  if (e.target.classList.contains("why")) set(e.target.dataset.id, {{why: e.target.value}});
}});
document.addEventListener("input", e => {{
  // 메모는 글자마다 다시 그리면 포커스를 잃는다. 저장만 하고 화면은 그대로 둔다.
  if (!e.target.classList.contains("memo")) return;
  marks[e.target.dataset.id] =
    Object.assign({{}}, marks[e.target.dataset.id], {{memo: e.target.value}});
  save();
}});
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (CHOICE[e.key]) {{ set(DATA.items[cur].id, {{pick: CHOICE[e.key]}}); goto(cur + 1); }}
  else if (e.key === "j" || e.key === "k") {{ goto(cur + (e.key === "j" ? 1 : -1)); render(); }}
}});

// 가림 상태도 남긴다. 37건을 한 번에 앉아서 볼 수 없으므로, 창을 다시 열었을 때 조건이
// 바뀌어 있으면 앞뒤 판단의 기준이 달라진다.
const OPEN_KEY = KEY + "-open";
document.getElementById("open").onchange = e => {{
  document.body.classList.toggle("open", e.target.checked);
  localStorage.setItem(OPEN_KEY, e.target.checked ? "1" : "0");
}};

function dump(){{
  const rows = DATA.items.map(it => {{
    const m = marks[it.id] || {{}};
    // 내보낼 때는 가림을 푼다. A·B 로만 남기면 나중에 어느 실험이 이겼는지 못 센다.
    const pick = m.pick || "";
    const won = pick === "A" || pick === "B" ? runOf(it, pick) : pick;
    return {{id: it.id, A: runOf(it, "A"), B: runOf(it, "B"),
            교사판정: it.teacher.judgement,
            A라벨같음: it.A.labelSame ? 1 : 0, B라벨같음: it.B.labelSame ? 1 : 0,
            고른것: pick, 이긴쪽: won,
            이유: m.why || "", 메모: (m.memo || "").replace(/\\t/g, " ")}};
  }});
  document.getElementById("out").value = mode === "json"
    ? JSON.stringify(rows, null, 1)
    : [Object.keys(rows[0]).join("\\t")]
        .concat(rows.map(r => Object.values(r).join("\\t"))).join("\\n");
}}
document.getElementById("tsv").onclick = () => {{ mode = "tsv"; dump(); }};
document.getElementById("json").onclick = () => {{ mode = "json"; dump(); }};
// 브라우저 저장소만으로는 부족하다. 저장소를 비우거나 다른 기기에서 열면 사라진다.
document.getElementById("download").onclick = () => {{
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const blob = new Blob([document.getElementById("out").value],
    {{type: mode === "json" ? "application/json" : "text/tab-separated-values"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `sentence-compare__${{DATA.a.run}}__${{DATA.b.run}}__${{stamp}}.${{
    mode === "json" ? "json" : "tsv"}}`;
  link.click();
  URL.revokeObjectURL(link.href);
}};
document.getElementById("reset").onclick = () => {{
  if (confirm("채점을 전부 지웁니다. 계속할까요?")) {{ marks = {{}}; save(); render(); }}
}};

if (localStorage.getItem(OPEN_KEY) === "1") {{
  document.getElementById("open").checked = true;
  document.body.classList.add("open");
}}
render();
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, type=Path, help="첫째 실험 폴더")
    ap.add_argument("--b", required=True, type=Path, help="둘째 실험 폴더")
    ap.add_argument("--label", nargs=2, default=None,
                    help="두 실험에 붙일 이름. 안 주면 폴더 이름을 쓴다")
    ap.add_argument("--data", default=HOLDOUT, help="교사 정답이 든 홀드아웃")
    ap.add_argument("--out", type=Path, default=None, help="안 주면 runs/문장채점.html")
    args = ap.parse_args()

    a_dir = args.a if args.a.is_absolute() else ROOT / args.a
    b_dir = args.b if args.b.is_absolute() else ROOT / args.b
    labels = tuple(args.label) if args.label else (a_dir.name, b_dir.name)

    html_text, stats = build(a_dir, b_dir, labels, ROOT / args.data)
    out = args.out or (ROOT / "runs" / "문장채점.html")
    out.write_text(html_text, encoding="utf-8")

    print(f"홀드아웃 {stats['총']}건")
    print(f"  A {labels[0]}  ({a_dir.name})")
    print(f"  B {labels[1]}  ({b_dir.name})")
    if stats["둘다빈문장"]:
        print(f"\n  · 둘 다 문장을 안 낸 것 {stats['둘다빈문장']}건 -- negative 판정이라 "
              "문장이 없는 경우가 대부분입니다.")
        print("    견줄 것이 없으므로 '둘이 같은 뜻'으로 넘기시면 됩니다.")
    print(f"\n저장: {out}")
    print("  자리(A·B)는 id로 정해지므로 다시 만들어도 같습니다 -- 채점이 딴 문장에 안 붙습니다.")


if __name__ == "__main__":
    main()
