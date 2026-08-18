"""문장 두 개를 **눈가림으로 맞붙여 사람이 채점하는 판**을 만든다. 판이 둘이다.

**이 저장소의 채점기는 문장을 못 잰다.** AM 다섯은 형식 검사, `교사일치`는 `judgement`
한 칸, `라벨일치`는 `(대상, 방향)` 집합이다. 셋 다 산출물인 `direct_impact`를 안 본다.
그래서 사람이 직접 읽고 고른다.

    teacher  Q1 -- 라벨이 교사와 달라도 문장은 같은 뜻인가
    rounds   Q2 -- 데이터를 늘려 문장이 좋아졌나

**둘 다 "문장 두 개를 놓고 하나를 고른다"로 같은 모양이다.** 다른 것은 무엇과 무엇을
짝지우느냐뿐이다 -- Q1은 모델과 교사, Q2는 1차 모델과 2차 모델이다.

## 가리는 것 셋

1. **어느 자리가 무엇인지.** Q1에서 교사인 줄 알면 교사 쪽에 점수를 준다. Q2에서 2차인
   줄 알면 "데이터가 많으니 낫겠지"로 기운다. 자리는 `id`와 출처로 정해지므로 **다시
   만들어도 같다** -- 난수면 판을 다시 만들 때 채점이 딴 문장에 붙는다.
2. **라벨이 교사와 맞았는지.** Q1이 재려는 것이 바로 "라벨 일치와 문장 일치의 관계"라,
   보여주면 그 관계를 사람이 만들어 버린다.
3. **교사 해석 전체.** Q2에서 교사 문장을 보여주면 "교사에 가까운 쪽"으로 쏠리는데,
   **교사는 정답이 아니라 교사 모델의 출력이다.** 조문만 보고 고르게 한다.

## Q1은 대조군이 필요하다

"라벨이 다른 건의 60%가 문장은 같더라"만으로는 아무 말도 못 한다 -- 라벨이 **같은** 건이
95%일 수도 있기 때문이다. 그래서 **라벨이 같은 건과 다른 건을 같은 수로 섞어 넣고,
어느 쪽인지 가린다.** 채점이 끝나면 두 무리의 비율을 견준다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.compare teacher --run runs/delora-r2 runs/miss-r2 --n 60 \\
        --out runs/Q1_라벨과문장.html
    python -m cli.compare rounds --pair runs/delora:runs/delora-r2 \\
        --out runs/Q2_데이터효과.html
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

# 4지선다. 하나만 고르면 두 가지가 동시에 답해진다 -- `같음` 비율이 "문장이 같은 뜻인가"고,
# `A`/`B` 승패가 "어느 쪽이 나은가"다.
CHOICES = {"1": "같음", "2": "A", "3": "B", "4": "둘다틀림"}


def digest(*parts: str) -> int:
    """문자열들로 정해지는 수. **난수를 안 쓴다** -- 판을 다시 만들어도 자리가 같아야 한다."""
    return int(hashlib.md5("\x1f".join(parts).encode()).hexdigest(), 16)


def sentence(row: dict) -> str:
    return (row.get("direct_impact") or "").strip()


def balance(rows: list[dict]) -> None:
    """`flip`을 **정확히 반씩** 매긴다. `id` 해시만 쓰면 한쪽으로 기운다.

    실측으로 80건에서 교사가 A자리에 48번(60%) 왔다. 80건을 채점하다 보면 "A가 대체로
    교사구나"를 눈치채고, 그러면 가린 뜻이 없어진다. 해시로 **줄을 세운 뒤 앞 절반만**
    뒤집으면 정확히 반씩이면서 자리에 규칙이 안 보인다.
    """
    rows.sort(key=lambda r: digest("seat", r["rid"], r["first"][0]))
    for i, row in enumerate(rows):
        row["flip"] = i < len(rows) // 2


def make_item(rid: str, teacher: dict, first: tuple[str, dict], second: tuple[str, dict],
              extra: dict, flip: bool) -> dict:
    """한 카드. `first`·`second`는 `(출처 이름, 레코드)`이고 자리는 `balance()`가 정한다."""
    left, right = (second, first) if flip else (first, second)
    before, after = diff_html(teacher.get("before", ""), teacher.get("after", ""))
    return {
        "id": rid, "short": teacher.get("before_id", rid).split("-")[-1],
        "beforeHtml": before, "afterHtml": after,
        "A": {"sentence": sentence(left[1])},
        "B": {"sentence": sentence(right[1])},
        # 가림을 풀거나 내보낼 때만 쓴다. 화면에는 안 뿌린다.
        "key": {"A": left[0], "B": right[0], **extra},
    }


def build_teacher(runs: list[Path], holdout: Path, n: int) -> tuple[list[dict], dict]:
    """Q1 -- 모델 문장과 교사 문장을 맞붙인다. **라벨 일치 여부로 반씩 뽑는다.**"""
    T = read_jsonl(holdout)
    pool: dict[str, list[dict]] = {"같음": [], "다름": []}
    for run in runs:
        R = read_jsonl(run / "eval" / "records.jsonl")
        for rid in T:
            if rid not in R:
                continue
            # 한쪽이라도 문장이 없으면 견줄 것이 없다. negative 건이 대부분이다.
            if not sentence(T[rid]) or not sentence(R[rid]):
                continue
            same = label_set(R[rid].get("labels")) == label_set(T[rid].get("labels"))
            pool["같음" if same else "다름"].append(
                {"rid": rid, "teacher": T[rid], "first": (run.name, R[rid]),
                 "second": ("교사", T[rid]),
                 "extra": {"labelSame": same, "run": run.name}})

    # 같은 조문이 실험마다 나오므로 그대로 쓰면 한 조문을 여러 번 읽는다. 무리마다
    # 조문이 고루 퍼지도록 섞은 뒤 앞에서 잘라 낸다.
    rows = []
    for group in ("같음", "다름"):
        pool[group].sort(key=lambda r: digest(r["rid"], r["first"][0], group))
        rows += pool[group][: n // 2]
    balance(rows)
    items = [make_item(r["rid"], r["teacher"], r["first"], r["second"], r["extra"], r["flip"])
             for r in rows]
    # 같은 조문이 잇달아 나오지 않게 다시 섞는다.
    items.sort(key=lambda it: digest("order", it["id"], it["key"]["run"]))
    for i, it in enumerate(items):
        it["n"] = i
    return items, {"라벨같음 후보": len(pool["같음"]), "라벨다름 후보": len(pool["다름"]),
                   "뽑은 것": len(items)}


def build_rounds(pairs: list[tuple[Path, Path]], holdout: Path) -> tuple[list[dict], dict]:
    """Q2 -- 같은 조문에 1차 모델과 2차 모델이 낸 문장을 맞붙인다."""
    T = read_jsonl(holdout)
    picked = []
    for a_dir, b_dir in pairs:
        A, B = (read_jsonl(d / "eval" / "records.jsonl") for d in (a_dir, b_dir))
        for rid in T:
            if rid not in A or rid not in B:
                continue
            if not sentence(A[rid]) or not sentence(B[rid]):
                continue
            # 글자까지 같으면 고를 것이 없다.
            if sentence(A[rid]) == sentence(B[rid]):
                continue
            picked.append({"rid": rid, "teacher": T[rid],
                           "first": (a_dir.name, A[rid]), "second": (b_dir.name, B[rid]),
                           "extra": {"pair": f"{a_dir.name} vs {b_dir.name}"}})
    balance(picked)
    items = [make_item(r["rid"], r["teacher"], r["first"], r["second"], r["extra"], r["flip"])
             for r in picked]
    items.sort(key=lambda it: digest("order", it["id"], it["key"]["pair"]))
    for i, it in enumerate(items):
        it["n"] = i
    return items, {"짝": len(pairs), "뽑은 것": len(items)}


PAGES = {
    "teacher": {
        "title": "Q1 · 라벨과 문장",
        "lede": ("<b>같은 조문에 두 해석이 있다. 어느 쪽이 무엇인지는 가려져 있다.</b> "
                 "한쪽은 모델이 낸 문장이고 한쪽은 교사가 낸 문장인데, 자리는 건마다 다르다. "
                 "<b>라벨이 교사와 맞았는지도 가렸다</b> — 이 판이 재려는 것이 바로 라벨과 "
                 "문장의 관계라, 보여주면 그 관계를 사람이 만들어 버린다.<br><br>"
                 "<b>재는 것</b> — 라벨이 교사와 달라도 문장은 같은 뜻인가. "
                 "라벨이 <b>같은</b> 건도 같은 수로 섞여 있다. 그것이 대조군이다."),
        "cols": ["id", "run", "A", "B", "라벨같음", "고른것", "이유", "메모"],
    },
    "rounds": {
        "title": "Q2 · 데이터 효과",
        "lede": ("<b>같은 조문에 두 실험이 낸 문장이다. 어느 쪽이 어느 실험인지는 가려져 "
                 "있고 자리는 건마다 다르다.</b> 한쪽은 562건으로 배운 모델이고 한쪽은 "
                 "1,655건으로 배운 모델이다.<br><br>"
                 "<b>교사 문장은 안 보여준다</b> — 보여주면 교사에 가까운 쪽으로 쏠리는데, "
                 "<b>교사는 정답이 아니라 교사 모델의 출력이다.</b> 개정 전후만 보고 "
                 "어느 쪽이 이 개정을 더 정확히 설명하는지로 고른다."),
        "cols": ["id", "pair", "A", "B", "고른것", "이긴쪽", "이유", "메모"],
    },
}


def page(mode: str, items: list[dict], meta: str) -> str:
    spec = PAGES[mode]
    blob = json.dumps({"mode": mode, "items": items, "cols": spec["cols"]}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{spec['title']}</title>
<style>
  :root{{
    --bg:#f6f7f9; --card:#fff; --ink:#16191d; --muted:#666e7a; --line:#dfe3e8;
    --del:#fdecec; --delink:#a02020; --ins:#e8f6ec; --insink:#12692e;
    --accent:#2a5bd7; --a:#7b4fd1; --b:#0f7b6c;
  }}
  @media (prefers-color-scheme:dark){{
    :root{{
      --bg:#14161a; --card:#1c1f24; --ink:#e6e8ec; --muted:#98a0ab; --line:#2c3038;
      --del:#3b1f1f; --delink:#ff9a90; --ins:#17301f; --insink:#7fd39a;
      --accent:#7aa2f7; --a:#b18cf0; --b:#5fc8b6;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.65 -apple-system,"Segoe UI","Malgun Gothic",sans-serif}}
  header{{position:sticky;top:0;z-index:10;background:var(--card);
         border-bottom:1px solid var(--line);padding:12px 20px;
         display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
  h1{{font-size:15px;margin:0;font-weight:700}}
  .meta{{color:var(--muted);font-size:12.5px}}
  .prog{{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:700;font-size:13.5px}}
  button{{font:inherit;padding:5px 12px;border:1px solid var(--line);background:var(--card);
         color:var(--ink);border-radius:6px;cursor:pointer}}
  button:hover{{border-color:var(--accent);color:var(--accent)}}
  main{{max-width:1000px;margin:0 auto;padding:20px}}
  .lede{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:14px 18px;margin-bottom:16px;color:var(--muted);font-size:13.5px}}
  .lede b{{color:var(--ink)}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:16px 18px;margin-bottom:14px}}
  .card.done{{border-left:4px solid var(--insink)}}
  .card.cur{{box-shadow:0 0 0 2px var(--accent)}}
  .row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}}
  .num{{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums}}
  .lbl{{font-size:12px;color:var(--muted);margin:12px 0 4px;letter-spacing:.04em}}
  .txt{{font:14px/1.85 ui-monospace,"D2Coding",Consolas,monospace;
       white-space:pre-wrap;word-break:break-word;background:var(--bg);
       border:1px solid var(--line);border-radius:7px;padding:10px 12px}}
  del{{background:var(--del);color:var(--delink);text-decoration:line-through}}
  ins{{background:var(--ins);color:var(--insink);text-decoration:none}}
  .sent{{border:1px solid var(--line);border-radius:7px;padding:11px 13px;margin-top:8px;
        display:flex;gap:12px;align-items:baseline}}
  .sent .who{{flex:0 0 20px;font-weight:700;font-size:13px}}
  .sent.A{{border-left:3px solid var(--a)}} .sent.A .who{{color:var(--a)}}
  .sent.B{{border-left:3px solid var(--b)}} .sent.B .who{{color:var(--b)}}
  .sent p{{margin:0;flex:1}}
  .reveal{{font-size:12px;color:var(--muted);margin-left:8px}}
  body:not(.open) .reveal{{visibility:hidden}}
  .score{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px;
         border-top:1px solid var(--line);padding-top:12px}}
  .g{{display:flex;gap:6px;flex-wrap:wrap}}
  .score button.on{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  select,input[type=text]{{font:inherit;font-size:13px;padding:5px 9px;border-radius:6px;
    border:1px solid var(--line);background:var(--card);color:var(--ink)}}
  input.memo{{flex:1;min-width:140px}}
  textarea{{width:100%;height:160px;font:12.5px/1.6 ui-monospace,Consolas,monospace;
    border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--card);
    color:var(--ink);margin-top:10px}}
</style>
</head>
<body>
<header>
  <h1>{spec['title']}</h1>
  <span class="meta">{html.escape(meta)}</span>
  <label class="meta"><input type="checkbox" id="open"> 가림 풀기</label>
  <button id="tsv">TSV</button><button id="json">JSON</button>
  <button id="download">파일로 저장</button><button id="reset">초기화</button>
  <span class="prog" id="prog"></span>
</header>
<main>
  <div class="lede">
    {spec['lede']}<br><br>
    <b>키보드</b> — <code>1</code> 둘이 같은 뜻 · <code>2</code> A가 낫다 ·
    <code>3</code> B가 낫다 · <code>4</code> 둘 다 틀렸다 ·
    <code>j</code>/<code>k</code> 이동. 고르면 다음 건으로 넘어간다.
    점수는 브라우저에 남지만 <b>끝나면 「파일로 저장」을 누를 것.</b>
  </div>
  <div id="list"></div>
  <textarea id="out" readonly></textarea>
</main>
<script>
const DATA = {blob};
const KEY = "compare-" + DATA.mode + "-" + DATA.items.length;
const LABEL = {{"1": "같음", "2": "A", "3": "B", "4": "둘다틀림"}};
const TEXT = {{"1": "둘이 같은 뜻", "2": "A가 낫다", "3": "B가 낫다", "4": "둘 다 틀렸다"}};
// 왜 그렇게 봤는지. 다음 판에서 무엇을 고쳐야 하는지는 이 분포에서 나온다.
const WHY = ["", "한쪽이 사실을 지어냄", "한쪽이 방향을 반대로 읽음", "한쪽이 덜 구체적",
             "한쪽만 주체를 빠뜨림", "뜻은 같고 말만 다름", "비문", "기타"];

let marks = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let cur = 0, mode = "tsv";

function esc(s){{
  return (s == null ? "" : String(s)).replace(/[&<>]/g,
    c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
}}
function attr(s){{ return esc(s).replace(/"/g, "&quot;"); }}
function markKey(it){{ return it.id + "|" + (it.key.run || it.key.pair || ""); }}

function render(){{
  document.getElementById("list").innerHTML = DATA.items.map(it => {{
    const m = marks[markKey(it)] || {{}};
    const box = seat => `<div class="sent ${{seat}}"><span class="who">${{seat}}</span>
      <p>${{esc(it[seat].sentence)}}<span class="reveal">${{esc(it.key[seat])}}</span></p></div>`;
    return `
    <div class="card ${{m.pick ? "done" : ""}} ${{it.n === cur ? "cur" : ""}}" id="c${{it.n}}">
      <div class="row">
        <span class="num">#${{it.n + 1}} / ${{DATA.items.length}}</span>
        <span class="num">${{esc(it.short)}}</span>
        <span class="reveal num">${{esc(it.key.run || it.key.pair || "")}}${{
          it.key.labelSame === undefined ? "" : " · 라벨 " + (it.key.labelSame ? "같음" : "다름")}}</span>
      </div>
      <div class="lbl">개정 전</div><div class="txt">${{it.beforeHtml}}</div>
      <div class="lbl">개정 후</div><div class="txt">${{it.afterHtml}}</div>
      <div class="lbl">두 해석</div>
      ${{box("A")}}${{box("B")}}
      <div class="score">
        <span class="g">
          ${{["1","2","3","4"].map(k =>
            `<button data-k="${{attr(markKey(it))}}" data-v="${{LABEL[k]}}"
              class="${{m.pick === LABEL[k] ? "on" : ""}}">${{k}} ${{TEXT[k]}}</button>`).join("")}}
        </span>
        <select data-k="${{attr(markKey(it))}}" class="why">
          ${{WHY.map(w => `<option value="${{attr(w)}}" ${{m.why === w ? "selected" : ""}}>${{
            esc(w) || "왜 그렇게 봤나 —"}}</option>`).join("")}}
        </select>
        <input type="text" class="memo" data-k="${{attr(markKey(it))}}" placeholder="메모"
               value="${{attr(m.memo || "")}}">
      </div>
    </div>`;
  }}).join("");

  const done = DATA.items.filter(it => (marks[markKey(it)] || {{}}).pick);
  const n = p => done.filter(it => marks[markKey(it)].pick === p).length;
  document.getElementById("prog").textContent =
    `${{done.length}} / ${{DATA.items.length}}` + (done.length
      ? `  ·  같음 ${{n("같음")}}  A ${{n("A")}}  B ${{n("B")}}  둘다틀림 ${{n("둘다틀림")}}` : "");
  dump();
}}

function save(){{ localStorage.setItem(KEY, JSON.stringify(marks)); }}
function set(k, patch){{ marks[k] = Object.assign({{}}, marks[k], patch); save(); render(); }}
function goto(i){{
  cur = Math.max(0, Math.min(DATA.items.length - 1, i));
  document.getElementById("c" + cur)?.scrollIntoView({{block: "center", behavior: "smooth"}});
}}

document.addEventListener("click", e => {{
  const b = e.target.closest("button[data-k]");
  if (!b) return;
  set(b.dataset.k, {{pick: b.dataset.v}});
  goto(DATA.items.findIndex(it => markKey(it) === b.dataset.k) + 1);
}});
document.addEventListener("change", e => {{
  if (e.target.classList.contains("why")) set(e.target.dataset.k, {{why: e.target.value}});
}});
document.addEventListener("input", e => {{
  // 메모는 글자마다 다시 그리면 포커스를 잃는다. 저장만 하고 화면은 그대로 둔다.
  if (!e.target.classList.contains("memo")) return;
  marks[e.target.dataset.k] =
    Object.assign({{}}, marks[e.target.dataset.k], {{memo: e.target.value}});
  save();
}});
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (LABEL[e.key]) {{ set(markKey(DATA.items[cur]), {{pick: LABEL[e.key]}}); goto(cur + 1); }}
  else if (e.key === "j" || e.key === "k") {{ goto(cur + (e.key === "j" ? 1 : -1)); render(); }}
}});

// 가림 상태도 남긴다. 한 번에 앉아서 다 볼 수 없으므로, 창을 다시 열었을 때 조건이
// 바뀌어 있으면 앞뒤 판단의 기준이 달라진다.
const OPEN_KEY = KEY + "-open";
document.getElementById("open").onchange = e => {{
  document.body.classList.toggle("open", e.target.checked);
  localStorage.setItem(OPEN_KEY, e.target.checked ? "1" : "0");
}};

function dump(){{
  // 내보낼 때는 가림을 푼다. A·B 로만 남기면 나중에 어느 쪽이 이겼는지 못 센다.
  const rows = DATA.items.map(it => {{
    const m = marks[markKey(it)] || {{}}, pick = m.pick || "";
    const row = {{
      id: it.id, run: it.key.run || "", pair: it.key.pair || "",
      A: it.key.A, B: it.key.B,
      라벨같음: it.key.labelSame === undefined ? "" : (it.key.labelSame ? 1 : 0),
      고른것: pick,
      이긴쪽: pick === "A" ? it.key.A : pick === "B" ? it.key.B : pick,
      이유: m.why || "", 메모: (m.memo || "").replace(/\\t/g, " "),
    }};
    return Object.fromEntries(DATA.cols.map(c => [c, row[c]]));
  }});
  document.getElementById("out").value = mode === "json"
    ? JSON.stringify(rows, null, 1)
    : [DATA.cols.join("\\t")]
        .concat(rows.map(r => DATA.cols.map(c => r[c]).join("\\t"))).join("\\n");
}}
document.getElementById("tsv").onclick = () => {{ mode = "tsv"; dump(); }};
document.getElementById("json").onclick = () => {{ mode = "json"; dump(); }};
// 브라우저 저장소만으로는 부족하다. 저장소를 비우거나 다른 기기에서 열면 사라진다.
document.getElementById("download").onclick = () => {{
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const blob = new Blob([document.getElementById("out").value],
    {{type: mode === "json" ? "application/json" : "text/tab-separated-values"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${{DATA.mode}}__${{stamp}}.${{mode === "json" ? "json" : "tsv"}}`;
  a.click();
  URL.revokeObjectURL(a.href);
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
    ap.add_argument("mode", choices=["teacher", "rounds"],
                    help="teacher = Q1 (모델 대 교사) · rounds = Q2 (1차 대 2차)")
    ap.add_argument("--run", nargs="+", type=Path, default=[], help="teacher 모드: 실험 폴더들")
    ap.add_argument("--pair", nargs="+", default=[],
                    help="rounds 모드: '1차경로:2차경로' 꼴로 여럿")
    ap.add_argument("--n", type=int, default=60, help="teacher 모드에서 뽑을 건수 (반씩 나눈다)")
    ap.add_argument("--data", default=HOLDOUT, help="교사 정답이 든 홀드아웃")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    def resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    if args.mode == "teacher":
        if not args.run:
            ap.error("teacher 모드에는 --run 이 필요하다")
        items, stats = build_teacher([resolve(r) for r in args.run],
                                     resolve(args.data), args.n)
        meta = f"실험 {len(args.run)}개에서 {len(items)}건"
    else:
        if not args.pair:
            ap.error("rounds 모드에는 --pair 가 필요하다")
        pairs = []
        for spec in args.pair:
            if ":" not in spec:
                ap.error(f"--pair 는 '1차경로:2차경로' 꼴이어야 한다: {spec}")
            a, b = spec.split(":", 1)
            pairs.append((resolve(a), resolve(b)))
        items, stats = build_rounds(pairs, resolve(args.data))
        meta = f"짝 {len(pairs)}개에서 {len(items)}건"

    out = args.out or (ROOT / "runs" / f"채점_{args.mode}.html")
    out.write_text(page(args.mode, items, meta), encoding="utf-8")

    for k, v in stats.items():
        print(f"  {k:<12} {v}")
    print(f"\n저장: {out}")
    print("  자리(A·B)는 id 로 정해지므로 다시 만들어도 같습니다 -- 채점이 딴 문장에 안 붙습니다.")


if __name__ == "__main__":
    main()
