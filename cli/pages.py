"""정리용 페이지를 만든다. **지금은 1페이지(결과)만이고 2·3페이지가 뒤에 붙는다.**

페이지가 하는 말은 셋이다.

    1절  직접 넣어 보기 -- 서버에 띄운 여섯 모델에 물어본다 (`cli/serve.py`)
    2절  같은 조문에 학습 전·잘된 것·이상한 것이 각각 무엇이라 답했나 (교사 없이)
    3절  학습 데이터가 562 -> 1,655 -> 3,178건으로 늘 때 학습법들이 어떻게 움직였나

**2·3절은 GPU를 안 쓴다.** 이미 채점해 저장해 둔 기록(`runs/<실험>/eval-mof-motie/`)과
표(`runs/sweep-mof-motie.json`)만 읽는다. 1절만 살아 있는 서버가 필요하다 --
**서버가 없어도 나머지는 그대로 읽힌다.**

**판정을 저장된 칸에서 읽지 않고 원문(`raw`)에서 다시 매긴다.** 2026-08-27에 파서를
갈아 끼웠는데 그것이 `records.jsonl`에는 안 닿았다 -- 거기 든 `judgement`·`labels`는
옛 파서가 읽은 값이라 표와 어긋난다(예: 학습 전 KORMo가 저장된 칸으로는 판정 7건,
다시 매기면 81건이다). 표를 만든 `cli/rescore.py --reparse`와 같은 길로 간다.

**여섯 칸의 목록은 `cli/serve.py`에서 가져온다.** 페이지와 서버가 다른 모델을 가리키면
1절에서 누른 것과 2절에서 본 것이 어긋나므로, 목록은 한 군데에만 둔다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.pages
    python -m cli.pages --out visualizations/정리_1_결과.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

from cli.readout import diff_html, read_jsonl
from cli.serve import CATALOG
from sft.scoring import parse_output

ROOT = Path(__file__).resolve().parents[1]

HOLDOUT = ROOT / "data/20260821__annotate__v2.2-run2A/holdout.jsonl"
SWEEP = ROOT / "runs/sweep-mof-motie.json"
EVAL_DIR = "eval-mof-motie"
OUT = ROOT / "visualizations/정리_1_결과.html"

# 원문을 통째로 넣으면 페이지가 몇 MB가 된다. 학습 전 모델은 종료를 몰라 언제나
# 상한(768토큰)까지 이어 쓰므로 특히 길다. 접어 두는 칸이라 끝을 자른다.
RAW_CAP = 2600

# 홀드아웃 135건의 교사 판정 분포에서 나오는 두 바닥값이다. 이 근처 값은 실력이
# 아니라 붕괴다 -- 무조건 한쪽으로 답해도 그만큼은 맞는다.
FLOOR_NEGATIVE = 0.163
FLOOR_POSITIVE = 0.837

ROUNDS = [
    ("prerun", "prerun", "562건", "negative 45.9%"),
    ("1차run", "1차run", "1,655건", "negative 15.95%"),
    ("2차run", "2차run", "3,178건", "negative 9.8%"),
]

# **라운드마다 실험 이름이 바뀌었다.** 같은 학습법인데 prerun은 `lora-full-rules`,
# 2차run은 `lora-run2A`다. 2차run 설정을 새로 쓸 때 이름을 줄인 탓이고, 이 표가 없으면
# 같은 학습법이 서로 다른 계열로 갈라져 그래프에 선이 안 이어진다.
RENAME = {
    "lora": "lora-full-rules",
    "rslora": "lora-rslora",
    "pissa": "lora-pissa",
    "delora-sentence-e3": "delora-sentence",
}

# 그래프에 적을 이름과, 어느 축의 실험인지. 축을 나누는 이유는 **실패의 뜻이 다르기**
# 때문이다 -- 방법 축은 "KORMo에 붙나"를 묻고, 데이터 축은 "무엇을 먹이나"를 묻는다.
FAMILIES = {
    "delora":               ("DeLoRA", "방법"),
    "gralora":              ("GraLoRA", "방법"),
    "lily":                 ("LILY", "방법"),
    "miss":                 ("MISS", "방법"),
    "peanut":               ("PEANUT", "방법"),
    "lora-full-rules":      ("LoRA (기준 조건)", "방법"),
    "lora-rslora":          ("LoRA + rsLoRA", "방법"),
    "lora-pissa":           ("LoRA + PiSSA", "방법"),
    "qwen-delora":          ("Qwen × DeLoRA", "방법"),
    "delora-sentence":      ("DeLoRA · 문장만", "데이터"),
    "lora-sentence-rules":  ("LoRA · 문장만", "데이터"),
    "lora-sentence-bare":   ("LoRA · 문장만 · 규칙서 없음", "데이터"),
    "lora-noimpacts-rules": ("LoRA · impacts 뺌", "데이터"),
    "lora-noimpacts-bare":  ("LoRA · impacts 뺌 · 규칙서 없음", "데이터"),
    "lora-full-bare":       ("LoRA · 규칙서 없음", "데이터"),
    "lora-full-nonegative": ("LoRA · negative 뺌", "데이터"),
    "lora-full-down120":    ("LoRA · 계열당 120건", "데이터"),
    "lora-r8":              ("LoRA · rank 8", "매개변수"),
    "lora-r16":             ("LoRA · rank 16", "매개변수"),
    "lora-r64":             ("LoRA · rank 64", "매개변수"),
    "lora-r128":            ("LoRA · rank 128", "매개변수"),
    "lora-lr5e-5":          ("LoRA · 학습률 5e-5", "매개변수"),
    "lora-lr1e-4":          ("LoRA · 학습률 1e-4", "매개변수"),
    "lora-lr5e-4":          ("LoRA · 학습률 5e-4", "매개변수"),
}

METRICS = [
    ("교사일치", "교사일치", "판정 한 칸이 교사와 같은 비율. 붕괴를 알아보는 눈금이다"),
    ("평균", "AM 평균", "형식 검사 다섯의 평균. 내용이 아니라 모양만 본다"),
    ("라벨일치", "라벨일치", "(대상, 방향) 집합이 교사와 같은 비율. 문장은 안 본다"),
]


# ---------------------------------------------------------------- 자료 읽기

def split_run(name: str) -> tuple[str, str] | None:
    """실험 이름에서 (라운드, 계열)을 뽑는다. 그래프에 안 올릴 것은 None이다.

    빼는 것 셋이다. `-3shot`·`-prefill1`은 **같은 어댑터를 다르게 물어본 것**이라
    라운드가 아니고, `-e6`은 에폭 축이라 데이터 양과 무관하다. 라운드 축에 섞으면
    "데이터가 늘어서 달라졌다"가 아니게 된다.
    """
    if "-3shot" in name or "-prefill1" in name or "-e6" in name:
        return None
    if name.startswith("baseline"):
        return None
    if name.endswith("-r2"):
        round_name, core = "1차run", name[:-3]
    elif (matched := re.match(r"(.+)-run2[AB]$", name)):
        round_name, core = "2차run", matched.group(1)
    else:
        round_name, core = "prerun", name
    # seed는 라운드가 아니라 같은 조건의 다른 알이다. 계열로 묶어 점을 여럿 찍는다.
    core = re.sub(r"-s4[34]$", "", core)
    return round_name, RENAME.get(core, core)


def load_items() -> list[dict]:
    """홀드아웃 135건 + 여섯 칸의 답. **답은 원문에서 다시 매긴다.**"""
    holdout = read_jsonl(HOLDOUT)
    answers: dict[str, dict[str, dict]] = {}
    for entry in CATALOG:
        run = entry["run"] or "baseline-kormo-rules"
        path = ROOT / "runs" / run / EVAL_DIR / "records.jsonl"
        if not path.exists():
            raise SystemExit(f"기록이 없습니다: {path}\n"
                             f"  bash sync_runs.sh 로 서버에서 당겨오세요")
        answers[entry["key"]] = read_jsonl(path)

    items = []
    for item_id, row in holdout.items():
        before_html, after_html = diff_html(row["before"], row["after"])
        cards = {}
        for entry in CATALOG:
            record = answers[entry["key"]].get(item_id)
            if record is None:
                cards[entry["key"]] = None
                continue
            raw = record.get("raw") or ""
            parsed = parse_output(raw) or {}
            cards[entry["key"]] = {
                "j": str(parsed.get("judgement", "")).strip(),
                "l": label_pairs(parsed.get("labels")),
                "s": str(parsed.get("direct_impact") or "").strip(),
                "r": raw[:RAW_CAP] + ("\n…(뒤를 잘랐습니다)" if len(raw) > RAW_CAP else ""),
                "t": record.get("new_tokens"),
            }
        items.append({
            "id": item_id,
            "series": row.get("series", ""),
            "bh": before_html, "ah": after_html,
            "before": row["before"], "after": row["after"],
            "a": cards,
        })
    return items


def label_pairs(labels) -> list[list[str]]:
    """`(대상, 방향)`만 남긴다. `근거`는 자유 문장이라 칩으로 못 그린다."""
    out = []
    for entry in labels or []:
        if isinstance(entry, dict):
            out.append([str(entry.get("대상", "")), str(entry.get("방향", ""))])
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            out.append([str(entry[0]), str(entry[1])])
    return out


def curate(items: list[dict]) -> list[dict]:
    """「이것부터 보세요」 다섯 건. **교사 답을 안 쓰고 고른다.**

    페이지가 교사를 안 보여주기로 했으므로 고르는 근거도 교사에 두면 안 된다 --
    "교사가 negative인 건"으로 고르면 그 목록 자체가 답을 흘린다. 여섯 칸이 서로
    어떻게 갈렸나와 조문이 얼마나 바뀌었나만 본다.
    """
    def said(item, key):
        card = item["a"].get(key)
        return card["j"] if card else ""

    def changed(item):
        return sum(len(chunk) for chunk in re.findall(r"<(?:del|ins)>(.*?)</", item["bh"] + item["ah"]))

    picks: list[tuple[str, dict]] = []

    split_most = max(items, key=lambda i: len({said(i, e["key"]) for e in CATALOG}))
    picks.append(("여섯 칸의 답이 가장 크게 갈린 조문", split_most))

    good_pair = [i for i in items if said(i, "kormo-good") != said(i, "qwen-good")]
    if good_pair:
        picks.append(("잘된 둘(KORMo·Qwen)이 서로 다르게 답한 조문", good_pair[0]))

    tiny = min((i for i in items if changed(i) > 0), key=changed)
    picks.append(("글자 몇 개만 바뀐 조문", tiny))

    big = max(items, key=changed)
    picks.append(("크게 손본 조문", big))

    seen, out = set(), []
    for why, item in picks:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        out.append({"why": why, "id": item["id"]})

    # 「안 바뀌었다」로 본 건은 마지막에 고른다. 앞의 넷과 겹치면 그다음 것을 집는다 --
    # 다섯 칸을 다 채우려는 것이 아니라, **문장이 비어 있는 답도 정상**이라는 것을
    # 보여주는 자리가 하나는 있어야 하기 때문이다.
    for item in items:
        if item["id"] in seen:
            continue
        if said(item, "kormo-good") == said(item, "qwen-good") == "negative":
            out.append({"why": "잘된 둘이 나란히 「안 바뀌었다」고 본 조문",
                        "id": item["id"]})
            break
    return out


def load_series() -> dict:
    """라운드별 기울기 그래프에 쓸 자료. `runs/sweep-mof-motie.json`이 원본이다."""
    table = json.loads(SWEEP.read_text(encoding="utf-8"))
    families: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for name, row in table.items():
        split = split_run(name)
        if split is None:
            continue
        round_name, family = split
        if family not in FAMILIES:
            continue
        families[family][round_name].append({
            "run": name,
            "verdict": row.get("verdict"),
            "교사일치": row.get("교사일치"),
            "평균": row.get("평균"),
            "라벨일치": row.get("라벨일치"),
        })

    out = []
    for family, rounds in families.items():
        label, axis = FAMILIES[family]
        out.append({
            "key": family, "label": label, "axis": axis,
            "rounds": {name: rows for name, rows in rounds.items()},
            "spans": len(rounds),
        })
    order = {"방법": 0, "데이터": 1, "매개변수": 2}
    out.sort(key=lambda f: (order[f["axis"]], -f["spans"], f["label"]))
    return out


# ---------------------------------------------------------------- 페이지

CSS = """
:root{
  --bg:#fcfcfb; --card:#f4f3f0; --card2:#ebeae6; --fg:#101010; --fg2:#4e4d49;
  --mut:#84837d; --line:#dcdbd6; --line2:#c9c8c2;
  --ins:#116b4a; --insb:#d8f0e4; --del:#a3271f; --delb:#fadedc;
  --pos:#0d6b3f; --posb:#dcf0e6; --neg:#8a5a00; --negb:#f8ecd0;
  --none:#7a7975; --noneb:#e6e5e1; --accent:#1f5fa8; --warn:#a3271f;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"D2Coding",monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#161715; --card:#212220; --card2:#2a2b28; --fg:#f0efe9; --fg2:#c2c1b8;
  --mut:#8d8c85; --line:#35362f; --line2:#454640;
  --ins:#6fd6a5; --insb:#123f2c; --del:#ff9b91; --delb:#4a1c19;
  --pos:#6fd6a5; --posb:#123f2c; --neg:#e8c069; --negb:#3d2f10;
  --none:#8d8c85; --noneb:#2e2f2b; --accent:#78b0ee; --warn:#ff9b91;
}}
:root[data-theme="dark"]{
  --bg:#161715; --card:#212220; --card2:#2a2b28; --fg:#f0efe9; --fg2:#c2c1b8;
  --mut:#8d8c85; --line:#35362f; --line2:#454640;
  --ins:#6fd6a5; --insb:#123f2c; --del:#ff9b91; --delb:#4a1c19;
  --pos:#6fd6a5; --posb:#123f2c; --neg:#e8c069; --negb:#3d2f10;
  --none:#8d8c85; --noneb:#2e2f2b; --accent:#78b0ee; --warn:#ff9b91;
}
*{box-sizing:border-box}
body{margin:0;padding:0 0 6rem;background:var(--bg);color:var(--fg);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",
  "맑은 고딕","Noto Sans KR",sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1180px;margin:0 auto;padding:0 1.4rem}
p{color:var(--fg2);margin:.5rem 0}
b,strong{color:var(--fg)}
code{font-family:var(--mono);font-size:.87em;background:var(--card);
  padding:.06rem .32rem;border-radius:4px}
a{color:var(--accent)}

nav{position:sticky;top:0;z-index:20;background:var(--bg);
  border-bottom:1px solid var(--line);margin-bottom:2rem}
nav .in{max-width:1180px;margin:0 auto;padding:.7rem 1.4rem;display:flex;
  gap:.5rem;align-items:center;flex-wrap:wrap}
nav .t{font-weight:700;margin-right:.6rem}
nav a,nav span.off{font-size:.87rem;padding:.28rem .7rem;border-radius:999px;
  text-decoration:none;border:1px solid var(--line)}
nav a.here{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:600}
nav a:not(.here){color:var(--fg2)}
nav span.off{color:var(--mut);border-style:dashed}

header.top{padding-top:.6rem}
h1{font-size:1.7rem;margin:0 0 .5rem;letter-spacing:-.01em}
h2{font-size:1.22rem;margin:3.4rem 0 .3rem;padding-top:1.1rem;
  border-top:2px solid var(--line)}
h3{font-size:1rem;margin:1.8rem 0 .3rem}
.lead{color:var(--fg);font-size:1.02rem;max-width:66ch}
.note{font-size:.87rem;color:var(--mut);max-width:74ch}
.sub{font-size:.9rem;color:var(--fg2);max-width:74ch}

.facts{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--line);
  border-radius:8px;overflow:hidden;margin:1.2rem 0;background:var(--card)}
.facts div{flex:1 1 150px;padding:.7rem .95rem;border-right:1px solid var(--line)}
.facts div:last-child{border-right:0}
.facts dt{font-size:.74rem;letter-spacing:.06em;color:var(--mut);
  font-family:var(--mono);margin:0 0 .15rem}
.facts dd{margin:0;font-size:1.12rem;font-weight:650;
  font-variant-numeric:tabular-nums}
.facts small{display:block;font-size:.76rem;color:var(--mut);font-weight:400}

.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.9rem 0}
button.chip{font:inherit;font-size:.84rem;padding:.32rem .78rem;cursor:pointer;
  border-radius:999px;border:1px solid var(--line);background:var(--card);
  color:var(--fg2)}
button.chip:hover{border-color:var(--line2);color:var(--fg)}
button.chip[aria-pressed="true"]{background:var(--fg);color:var(--bg);
  border-color:var(--fg);font-weight:600}
select,input[type=number],textarea{font:inherit;background:var(--card);
  color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.4rem .55rem}
textarea{width:100%;font-size:.92rem;line-height:1.6;resize:vertical}
button.go{font:inherit;font-weight:600;padding:.5rem 1.1rem;border-radius:6px;
  border:1px solid var(--fg);background:var(--fg);color:var(--bg);cursor:pointer}
button.go:disabled{opacity:.45;cursor:not-allowed}

.pair{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin:.8rem 0 1.4rem}
.pair>div{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:.7rem .9rem;font-size:.92rem;line-height:1.72;overflow-wrap:anywhere}
.pair h4{margin:0 0 .4rem;font-size:.76rem;letter-spacing:.06em;color:var(--mut);
  font-family:var(--mono);font-weight:600}
del{background:var(--delb);color:var(--del);text-decoration:none;
  border-radius:3px;padding:0 .1em}
ins{background:var(--insb);color:var(--ins);text-decoration:none;
  border-radius:3px;padding:0 .1em}
@media (max-width:800px){.pair{grid-template-columns:1fr}}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:.8rem;margin:1rem 0}
.mc{border:1px solid var(--line);border-radius:9px;background:var(--card);
  padding:.8rem .95rem;display:flex;flex-direction:column;gap:.5rem}
.mc.good{border-color:var(--line2)}
.mc .hd{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem}
.mc .nm{font-weight:650;font-size:.94rem}
.mc .rn{font-family:var(--mono);font-size:.74rem;color:var(--mut)}
.mc .note{font-size:.78rem;color:var(--mut);margin:0}
.mc .sent{font-size:1rem;line-height:1.6;color:var(--fg);
  border-left:3px solid var(--line2);padding-left:.7rem;min-height:1.2rem}
.mc .sent.empty{color:var(--mut);font-style:italic;border-left-color:var(--line)}
.badge{display:inline-block;font-size:.76rem;font-weight:650;padding:.1rem .5rem;
  border-radius:4px;font-family:var(--mono)}
.badge.positive{color:var(--pos);background:var(--posb)}
.badge.negative{color:var(--neg);background:var(--negb)}
.badge.none{color:var(--none);background:var(--noneb)}
.tag{display:inline-block;font-size:.76rem;padding:.1rem .45rem;border-radius:4px;
  background:var(--card2);color:var(--fg2);margin:0 .22rem .22rem 0}
.tag.none{color:var(--mut);background:transparent;border:1px dashed var(--line)}
details.raw summary{font-size:.78rem;color:var(--mut);cursor:pointer}
details.raw pre{font-family:var(--mono);font-size:.72rem;line-height:1.45;
  background:var(--bg);border:1px solid var(--line);border-radius:6px;
  padding:.55rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all;
  max-height:22rem;margin:.4rem 0 0}

.picker{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.7rem 0}
.picker select{min-width:min(30rem,100%)}

input[type=file]{font:inherit;font-size:.84rem;color:var(--fg2);max-width:100%}
button.chip:disabled{opacity:.32;cursor:not-allowed}
.rgn{border:1px solid var(--line);border-radius:7px;background:var(--card);
  padding:.5rem .7rem;margin:.4rem 0}
.rgn.off{opacity:.55}
.rgn label.hd{display:flex;gap:.45rem;align-items:baseline;cursor:pointer;
  font-family:var(--mono);font-size:.76rem;color:var(--mut)}
.pv{font-size:.88rem;line-height:1.65;overflow-wrap:anywhere;margin-top:.3rem}
.pv i{color:var(--mut);font-style:normal}

figure{margin:1.2rem 0}
figure svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.8rem;color:var(--mut);
  margin:.5rem 0}
.legend i{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;
  margin-right:.3rem;vertical-align:baseline}

.status{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;
  border:1px solid var(--line);border-radius:8px;background:var(--card);
  padding:.6rem .9rem;margin:1rem 0;font-size:.9rem}
.dot{width:.6rem;height:.6rem;border-radius:50%;background:var(--mut);
  display:inline-block;flex:none}
.dot.on{background:var(--ins)}
.dot.off{background:var(--del)}
pre.cmd{font-family:var(--mono);font-size:.78rem;background:var(--card);
  border:1px solid var(--line);border-radius:6px;padding:.6rem .75rem;
  overflow-x:auto;margin:.5rem 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}
@media (max-width:800px){.grid2{grid-template-columns:1fr}}
.adv{margin:.8rem 0;font-size:.86rem}
.adv summary{cursor:pointer;color:var(--mut)}
.adv .row{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin-top:.6rem}
label{font-size:.86rem;color:var(--fg2)}
table.t{border-collapse:collapse;width:100%;font-size:.87rem;margin:.8rem 0}
table.t th{text-align:left;font-weight:600;color:var(--fg2);font-size:.8rem;
  border-bottom:1px solid var(--line2);padding:.4rem .5rem}
table.t td{padding:.4rem .5rem;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;vertical-align:top}
"""


def nav_html(here: int) -> str:
    pages = [(1, "1 · 결과", "정리_1_결과.html"),
             (2, "2 · 데이터 수집", None),
             (3, "3 · 모델 학습", None)]
    parts = []
    for number, label, href in pages:
        if href is None:
            parts.append(f"<span class=off title='아직 안 만들었습니다'>{label}</span>")
        elif number == here:
            parts.append(f"<a class=here href='{href}'>{label}</a>")
        else:
            parts.append(f"<a href='{href}'>{label}</a>")
    return ("<nav><div class=in><span class=t>SLM 학습 프로젝트</span>"
            + "".join(parts) + "</div></nav>")


def data_island(name: str, payload) -> str:
    """JSON을 스크립트 칸에 넣는다. `</`를 막는 것은 **본문 안의 `</script>`가 태그로
    읽혀 페이지가 거기서 끊기는 것**을 피하려는 것이다."""
    text = json.dumps(payload, ensure_ascii=False).replace("</", r"<\/")
    return f"<script type='application/json' id='{name}'>{text}</script>"


def build_page1(items: list[dict], picks: list[dict], series: list[dict]) -> str:
    cards_meta = [{k: entry[k] for k in ("key", "label", "run", "note", "detail")}
                  for entry in CATALOG]
    options = "".join(
        f"<option value='{i}'>{i + 1:>3}. {html.escape(item['id'])}</option>"
        for i, item in enumerate(items))
    pick_buttons = "".join(
        f"<button class=chip data-pick='{html.escape(p['id'])}'>"
        f"{html.escape(p['why'])}</button>" for p in picks)
    metric_buttons = "".join(
        f"<button class=chip data-metric='{key}' title='{html.escape(why)}'"
        f"{' aria-pressed=true' if key == '교사일치' else ''}>{label}</button>"
        for key, label, why in METRICS)

    spans3 = sum(1 for f in series if f["spans"] == 3)
    return f"""<!doctype html>
<html lang=ko><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>결과 — 학습 전과 후, 그리고 잘못 배운 것</title>
<style>{CSS}</style></head><body>
{nav_html(1)}
<main>
<header class=top>
<h1>결과 — 학습 전과 후, 그리고 잘못 배운 것</h1>
<p class=lead>학습을 안 시킨 SLM(작은 언어모델)과 잘 학습된 것, 그리고 <b>잘못
학습된 것</b>이 같은 조문에 각각 무엇이라 답하는지를 나란히 놓는다. 먼저 직접 넣어 보고,
그다음 저장해 둔 응답 135건을 훑고, 마지막에 학습 데이터가 세 배씩 늘 때 학습법들이
어떻게 움직였는지를 본다.</p>
<p class=note><b>교사(정답을 만든 모델)의 답은 이 페이지 어디에도 안 나온다.</b>
어느 쪽이 더 그럴듯한지를 먼저 눈으로 정하고 나서 숫자를 보라는 뜻이다 — 2026-08-19에
사람이 눈가림으로 채점해 보니 모델 문장과 교사 문장이 거의 구분되지 않았다.</p>
<div class=facts>
<div><dt>채점 문항</dt><dd>135건<small>홀드아웃. 학습에 한 번도 안 쓴 조문</small></dd></div>
<div><dt>비교하는 모델</dt><dd>6개<small>KORMo 5 · Qwen 1</small></dd></div>
<div><dt>돌린 실험</dt><dd>174줄<small>같은 자로 다시 매긴 것 119줄</small></dd></div>
<div><dt>학습 데이터</dt><dd>562 → 3,178건<small>세 라운드</small></dd></div>
</div>
</header>

<h2>1. 직접 넣어 보기</h2>
<p class=sub>여섯 모델이 GPU에 떠 있으면 아래 칸이 살아난다. 개정 전후 조문을 넣고
누르면 여섯 칸이 각자 답한다. <b>답을 만드는 길은 채점 때와 한 글자도 다르지 않다</b> —
같은 프롬프트, 같은 프리필, 같은 greedy 생성이다. 그래서 여기서 나온 답을 아래 2절의
저장된 답과 나란히 놓아도 된다.</p>
<div class=status id=status>
<span class=dot id=dot></span><span id=statusText>서버를 찾는 중…</span>
<button class=chip id=retry>다시 찾기</button>
</div>
<div id=offline hidden>
<p class=note>서버가 안 잡힌다. <b>서버가 없어도 2·3절은 그대로 읽힌다</b> —
저장해 둔 기록이라 GPU를 안 쓴다.</p>
<p class=note>서버에서 아래를 띄운다. <code>--host 0.0.0.0</code>으로 띄우면 이 서버가
페이지까지 같이 내주므로, 보는 사람은 <code>http://&lt;서버주소&gt;:8000/</code>
하나만 열면 된다.</p>
<pre class=cmd>cd /data1/yblee/repository/model_train &amp;&amp; source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5 python -m cli.serve --host 0.0.0.0 --port 8000</pre>
<p class=note>혼자 볼 때는 <code>--host</code> 없이 띄우고 로컬(WSL)에서 굴을 판 뒤
「다시 찾기」를 누른다. 켜 둔 채로 둔다.</p>
<pre class=cmd>ssh -L 8000:localhost:8000 &lt;서버주소&gt;</pre>
</div>
<div class=picker>
<label for=loadFrom>홀드아웃에서 불러오기</label>
<select id=loadFrom><option value=''>— 직접 쓰기 —</option>{options}</select>
</div>
<div class=grid2>
<div><label for=inBefore>개정 전</label>
<textarea id=inBefore rows=7 placeholder='개정 전 조항을 붙여 넣으세요'></textarea></div>
<div><label for=inAfter>개정 후</label>
<textarea id=inAfter rows=7 placeholder='개정 후 조항을 붙여 넣으세요'></textarea></div>
</div>
<details class=adv><summary>고급 — 규칙서 · 프리필 · 길이 상한</summary>
<div class=row>
<label><input type=checkbox id=useRules checked> 역할 A 규칙서를 붙인다
<span class=note>(3,393자. 떼면 학습 전 모델은 대상 7종·방향 5종이라는 어휘 자체를 모른다)</span></label>
</div>
<div class=row>
<label>프리필 <input type=text id=prefill size=22 value='{{&#10;  "judgement": "'></label>
<label>max_new_tokens <input type=number id=maxTok value=768 min=64 max=2048 step=64></label>
</div>
</details>
<p><button class=go id=ask disabled>여섯 모델에 물어보기</button>
<span class=note id=askNote></span></p>
<div class=cards id=live></div>

<details class=adv id=fileBox><summary>한글파일(hwpx)로 넣기 — 문서 두 벌을 통째로 올린다</summary>
<p class=note>개정 전·후 문서를 올리면 <b>바뀐 곳만 찾아</b> 목록으로 보여 준다. 물어볼
곳을 고르고 모델 하나를 골라 누른다. 문서를 통째로 모델에 넣는 것이 아니다 —
<b>학습이 조문 한 개짜리로 되어 있어</b> 통째로 넣으면 길이가 열 배씩 넘친다.</p>
<p class=note><b>신형 hwpx만 읽는다.</b> 구형 .hwp는 속이 다른 형식이라 못 연다.
한글에서 「다른 이름으로 저장」으로 hwpx를 만들어 올린다.</p>
<div class=grid2>
<div><label for=fileBefore>개정 전 파일</label>
<input type=file id=fileBefore accept=".hwpx"></div>
<div><label for=fileAfter>개정 후 파일</label>
<input type=file id=fileAfter accept=".hwpx"></div>
</div>
<p><button class=go id=scan disabled>바뀐 곳 찾기</button>
<span class=note id=scanNote></span></p>
<div id=regionBox hidden>
<p class=note id=regionHead></p>
<div id=regions></div>
<p class=note>어느 모델에게 물을지 고른다. 규칙서·프리필·길이 상한은 위 「고급」 칸의
설정을 그대로 쓴다.</p>
<div class=chips id=modelPick></div>
<p><button class=go id=runFile disabled>요약하기</button>
<span class=note id=runNote></span></p>
<div class=cards id=fileLive></div>
</div>
</details>

<h2>2. 같은 조문, 여섯 가지 답</h2>
<p class=sub>홀드아웃 135건은 <b>mof 37건</b>(해양수산부 연구개발사업 운영규정)과
<b>motie 98건</b>(산업기술혁신사업 공통 운영요령)이다. 아래 답은 지금 만들어 내는 것이
아니라 <b>채점 때 실제로 나온 응답을 그대로 꺼낸 것</b>이다.</p>
<div class=chips>{pick_buttons}</div>
<div class=picker>
<label for=item>조문 고르기</label>
<select id=item>{options}</select>
<button class=chip id=prev>◀ 앞</button><button class=chip id=next>뒤 ▶</button>
</div>
<div class=pair>
<div><h4>개정 전</h4><div id=before></div></div>
<div><h4>개정 후</h4><div id=after></div></div>
</div>
<p class=note>색이 있는 자리가 바뀐 곳이다. <del>지워진 것</del>과 <ins>더해진 것</ins>을
글자 단위로 갈랐다 — 낱말 단위로 자르면 조사 하나 바뀐 것이 낱말 통째로 바뀐 것처럼 보인다.</p>
<div class=cards id=cards></div>

<h2>3. 데이터가 늘 때 학습법들이 어떻게 움직였나</h2>
<p class=sub>세 라운드 사이에 바뀐 것은 <b>학습 데이터 하나뿐이다.</b> 방법도 학습률도
rank도 seed도 에폭(3)도 그대로다. 그래서 선이 오르내린 것을 데이터 탓으로 읽을 수 있다.
가로축은 학습 데이터 양, 세로축은 고른 눈금이다. 세 라운드가 다 있는 <b>{spans3}계열</b>을
그린다 — 두 라운드까지만 간 조건들은 1차run에서 갈래가 정해져 안 데려간 것이다.</p>
<div class=chips>{metric_buttons}</div>
<p class=note><b>눈금 셋은 셋 다 <code>direct_impact</code> 문장을 안 본다.</b>
「AM 평균」은 형식만, 「교사일치」는 <code>judgement</code> 한 칸만, 「라벨일치」는
<code>(대상, 방향)</code> 집합만 본다. 이 과제의 산출물은 문장이므로 <b>세 눈금은 전부
그 앞 단계</b>다 — 라벨이 달라도 문장은 얼추 같은 경우가 흔하다. 문장을 보려면 2절로
돌아가 직접 읽는다.</p>
<figure id=big></figure>
<div class=legend>
<span><i style='background:var(--pos)'></i>됨</span>
<span><i style='background:var(--neg)'></i>이상함</span>
<span><i style='background:var(--del)'></i>붕괴</span>
<span>― 선은 그 라운드의 <b>중앙값</b> (seed·조건이 여럿이면 점을 다 찍는다)</span>
</div>
<p class=note><b>바닥선 두 줄을 같이 본다.</b> 홀드아웃 135건이 positive 113 ·
negative 22이라, 무조건 negative라고만 답해도 <b>16.3%</b>, 무조건 positive면
<b>83.7%</b>가 나온다. 이 두 값 근처는 실력이 아니라 붕괴다.</p>
<p class=note><b>라운드마다 실험 이름이 바뀐 것이 있다.</b> 같은 학습법인데 prerun은
<code>lora-full-rules</code>, 2차run은 <code>lora-run2A</code>다 — 2차run 설정을 새로 쓰며
이름을 줄인 탓이다. 그래프는 <code>cli/pages.py</code>의 대응표로 이어 붙인 것이다.
<code>-3shot</code>·<code>-prefill1</code>은 같은 어댑터를 다르게 물어본 것이라,
<code>-e6</code>는 에폭 축이라 여기서 뺐다.</p>
</main>

{data_island('ITEMS', items)}
{data_island('PICKS', picks)}
{data_island('CARDS', cards_meta)}
{data_island('SERIES', series)}
{data_island('ROUNDS', [list(r) for r in ROUNDS])}
<script>
{JS}
</script>
</body></html>
"""


JS = r"""
const ITEMS  = JSON.parse(document.getElementById('ITEMS').textContent);
const PICKS  = JSON.parse(document.getElementById('PICKS').textContent);
const CARDS  = JSON.parse(document.getElementById('CARDS').textContent);
const SERIES = JSON.parse(document.getElementById('SERIES').textContent);
const ROUNDS = JSON.parse(document.getElementById('ROUNDS').textContent);
const FLOORS = {neg: 0.163, pos: 0.837};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => (
  {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* ------------------------------------------------ 2절 · 저장된 답 */

function judgeBadge(j) {
  if (j === 'positive') return "<span class='badge positive'>positive</span>";
  if (j === 'negative') return "<span class='badge negative'>negative</span>";
  return "<span class='badge none'>못 읽음</span>";
}

function answerCard(meta, ans) {
  if (!ans) return `<div class=mc><div class=nm>${esc(meta.label)}</div>
    <p class=note>이 조문의 기록이 없습니다</p></div>`;
  const tags = ans.l.length
    ? ans.l.map((p) => `<span class=tag>${esc(p[0])} · ${esc(p[1])}</span>`).join('')
    : "<span class='tag none'>라벨 없음</span>";
  const why = ans.j === 'negative'
    ? '문장 없음 — 「안 바뀌었다」고 본 것이라 쓸 문장이 없다' : '문장 없음';
  const sentence = ans.s
    ? `<div class=sent>${esc(ans.s)}</div>`
    : `<div class='sent empty'>${why}</div>`;
  const cls = meta.key.includes('good') ? 'mc good' : 'mc';
  const tokens = (ans.t === null || ans.t === undefined) ? '?' : ans.t;
  return `<div class='${cls}'>
    <div class=hd><span class=nm>${esc(meta.label)}</span>
      <span class=rn>${esc(meta.run || '어댑터 없음')}</span></div>
    <p class=note>${esc(meta.note)}</p>
    <div>${judgeBadge(ans.j)} <span class=note>${tokens}토큰</span></div>
    ${sentence}
    <div>${tags}</div>
    <details class=raw><summary>모델이 실제로 뱉은 것</summary>
      <pre>${esc(ans.r)}</pre></details>
  </div>`;
}

let current = 0;
function renderItem(index) {
  current = Math.max(0, Math.min(ITEMS.length - 1, index));
  const item = ITEMS[current];
  $('item').value = String(current);
  $('before').innerHTML = item.bh;
  $('after').innerHTML = item.ah;
  $('cards').innerHTML = CARDS.map((m) => answerCard(m, item.a[m.key])).join('');
  document.querySelectorAll('[data-pick]').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.pick === item.id)));
}
$('item').addEventListener('change', (e) => renderItem(+e.target.value));
$('prev').addEventListener('click', () => renderItem(current - 1));
$('next').addEventListener('click', () => renderItem(current + 1));
document.querySelectorAll('[data-pick]').forEach((b) =>
  b.addEventListener('click', () =>
    renderItem(ITEMS.findIndex((i) => i.id === b.dataset.pick))));

/* ------------------------------------------------ 3절 · 라운드 그래프 */

const V_COLOR = {'됨': 'var(--pos)', '이상함': 'var(--neg)', '붕괴': 'var(--del)'};
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const h = Math.floor(s.length / 2);
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};

function points(family, metric) {
  return ROUNDS.map(([key], i) => {
    const rows = (family.rounds[key] || [])
      .filter((r) => typeof r[metric] === 'number');
    return {i, key, rows, mid: rows.length ? median(rows.map((r) => r[metric])) : null};
  });
}

function drawAll(metric) {
  const three = SERIES.filter((f) => f.spans === 3);
  const w = 900, h = 420;
  const padL = 46, padR = 120, padT = 14, padB = 34;
  const X = (i) => padL + (w - padL - padR) * (i / (ROUNDS.length - 1));
  const Y = (v) => padT + (h - padT - padB) * (1 - v);
  let svg = '';
  for (let v = 0; v <= 1.0001; v += 0.25) {
    svg += `<line x1=${padL} y1=${Y(v)} x2=${w - padR} y2=${Y(v)}
      stroke='var(--line)' stroke-width=1 />
      <text x=${padL - 6} y=${Y(v) + 3.5} font-size=10 text-anchor=end
      fill='var(--mut)'>${Math.round(v * 100)}</text>`;
  }
  if (metric === '교사일치') {
    for (const [v, txt] of [[FLOORS.neg, '무조건 negative 16.3%'],
                            [FLOORS.pos, '무조건 positive 83.7%']]) {
      svg += `<line x1=${padL} y1=${Y(v)} x2=${w - padR} y2=${Y(v)}
        stroke='var(--line2)' stroke-dasharray='3 3' />
        <text x=${w - padR + 6} y=${Y(v) + 3.5} font-size=10
        fill='var(--mut)'>${txt}</text>`;
    }
  }
  const ends = [];
  for (const family of three) {
    const cols = points(family, metric).filter((c) => c.mid !== null);
    if (!cols.length) continue;
    let d = '';
    cols.forEach((c, k) => { d += `${k ? 'L' : 'M'}${X(c.i)},${Y(c.mid)} `; });
    svg += `<path d='${d}' fill=none stroke='var(--fg2)' stroke-width=1.6
      stroke-opacity=.55 />`;
    for (const c of cols) for (const r of c.rows) {
      svg += `<circle cx=${X(c.i)} cy=${Y(r[metric])} r=4
        fill='${V_COLOR[r.verdict] || 'var(--mut)'}' fill-opacity=.85>
        <title>${esc(r.run)} — ${esc(r.verdict)} — ${(r[metric] * 100).toFixed(1)}%</title>
        </circle>`;
    }
    ends.push({y: Y(cols[cols.length - 1].mid), label: family.label,
               x: X(cols[cols.length - 1].i)});
  }
  ends.sort((a, b) => a.y - b.y);
  let lastY = -99;
  for (const e of ends) {
    const y = Math.max(e.y, lastY + 12);
    lastY = y;
    svg += `<text x=${e.x + 8} y=${y + 3.5} font-size=10.5
      fill='var(--fg)'>${esc(e.label)}</text>`;
  }
  ROUNDS.forEach(([key, name, size, mix], i) => {
    svg += `<text x=${X(i)} y=${h - padB + 15} font-size=11 text-anchor=middle
      fill='var(--fg)'>${esc(name)}</text>
      <text x=${X(i)} y=${h - padB + 28} font-size=10 text-anchor=middle
      fill='var(--mut)'>${esc(size)} · ${esc(mix)}</text>`;
  });
  $('big').innerHTML = `<svg viewBox='0 0 ${w} ${h}' role=img>${svg}</svg>`;

}

document.querySelectorAll('[data-metric]').forEach((b) =>
  b.addEventListener('click', () => {
    document.querySelectorAll('[data-metric]').forEach((o) =>
      o.setAttribute('aria-pressed', String(o === b)));
    drawAll(b.dataset.metric);
  }));

/* ------------------------------------------------ 1절 · 실습 */

/* 서버 주소를 박아 두지 않는다. 서버가 이 페이지를 직접 내주면(GET /) 페이지와
   답이 같은 출처라 자기가 실려 온 주소로 물으면 되고, HTML 파일을 그냥 열었으면
   (file://) 출처가 없으므로 굴 저편의 localhost로 묻는다. 박아 두면 남이 이 페이지를
   열었을 때 **자기** localhost에 묻게 되어 아무것도 안 잡힌다. */
const API = location.protocol.startsWith('http')
  ? location.origin : 'http://localhost:8000';
let ready = new Set();

async function probe() {
  $('dot').className = 'dot';
  $('statusText').textContent = '서버를 찾는 중…';
  try {
    const res = await fetch(API + '/models', {cache: 'no-store'});
    const data = await res.json();
    ready = new Set(data.models.filter((m) => m.ready).map((m) => m.key));
    const bad = data.models.filter((m) => !m.ready);
    $('dot').className = ready.size ? 'dot on' : 'dot off';
    $('statusText').innerHTML = `${ready.size}/${data.models.length}칸 준비됨` +
      (bad.length ? ` — 못 뜬 칸: ${bad.map((m) => esc(m.label)).join(', ')}` : '');
    $('offline').hidden = ready.size > 0;
    $('ask').disabled = ready.size === 0;
  } catch (err) {
    ready = new Set();
    $('dot').className = 'dot off';
    $('statusText').textContent = '서버가 안 잡힌다';
    $('offline').hidden = false;
    $('ask').disabled = true;
  }
  // 파일 칸의 모델 토글도 같은 `ready`를 본다. 서버가 늦게 떠도 「다시 찾기」 한 번에
  // 두 곳이 같이 살아난다.
  paintModels();
}
$('retry').addEventListener('click', probe);

$('loadFrom').addEventListener('change', (e) => {
  if (e.target.value === '') return;
  const item = ITEMS[+e.target.value];
  $('inBefore').value = item.before;
  $('inAfter').value = item.after;
});

function liveBody(state) {
  if (state.pending) return "<div class='sent empty'>묻는 중…</div>";
  if (state.error) {
    return `<div class='sent empty' style='color:var(--warn)'>` +
      esc(state.error) + `</div>`;
  }
  const parsed = state.parsed || {};
  const labels = (parsed.labels || []).map((l) =>
    `<span class=tag>${esc(l['대상'])} · ${esc(l['방향'])}</span>`).join('')
    || "<span class='tag none'>라벨 없음</span>";
  const sentence = String(parsed.direct_impact || '').trim();
  const sentenceClass = sentence ? 'sent' : 'sent empty';
  return `<div>${judgeBadge(String(parsed.judgement || '').trim())}
      <span class=note>${state.new_tokens}토큰 · ${state.seconds}초</span></div>
    <div class='${sentenceClass}'>${sentence ? esc(sentence) : '문장 없음'}</div>
    <div>${labels}</div>
    <details class=raw><summary>모델이 실제로 뱉은 것</summary>
      <pre>${esc(state.raw)}</pre></details>`;
}

function liveCard(meta, state) {
  const cls = meta.key.includes('good') ? 'mc good' : 'mc';
  return `<div class='${cls}'>
    <div class=hd><span class=nm>${esc(meta.label)}</span>
      <span class=rn>${esc(meta.run || '어댑터 없음')}</span></div>
    <p class=note>${esc(meta.note)}</p>${liveBody(state)}</div>`;
}

$('ask').addEventListener('click', async () => {
  const before = $('inBefore').value.trim(), after = $('inAfter').value.trim();
  if (!before || !after) { $('askNote').textContent = '두 칸을 다 채우세요'; return; }
  $('askNote').textContent = '';
  $('ask').disabled = true;

  const targets = CARDS.filter((m) => ready.has(m.key));
  const states = {};
  targets.forEach((m) => { states[m.key] = {pending: true}; });
  const paint = () => {
    $('live').innerHTML = targets.map((m) => liveCard(m, states[m.key])).join('');
  };
  paint();

  const body = {
    before, after, rules: $('useRules').checked,
    prefill: $('prefill').value,
    max_new_tokens: +$('maxTok').value,
  };
  // 하나씩 보낸다. 서버가 GPU마다 자물쇠를 걸어 어차피 줄을 서므로, 한꺼번에 던지면
  // 먼저 온 칸이 먼저 그려지는 것만 잃는다.
  for (const meta of targets) {
    try {
      const res = await fetch(API + '/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...body, model: meta.key}),
      });
      const data = await res.json();
      states[meta.key] = data.ok ? data : {error: data.error || '알 수 없는 오류'};
    } catch (err) {
      states[meta.key] = {error: String(err)};
    }
    paint();
  }
  $('ask').disabled = false;
});

/* ------------------------------------- 1절 · 한글파일(hwpx)로 넣기 */

// 한 건에 걸리는 시간(초). `runs/<실험>/eval-mof-motie/summary.json`의
// elapsed_seconds를 135로 나눈 실측이다. **미리 안 적으면 멈춘 줄 알고
// 새로고침한다** -- 21군데짜리 문서는 KORMo로도 6분, Qwen이면 14분이다.
//
// 여기는 줄 주석을 쓴다. 경로에 별표를 넣으면 별표와 빗금이 붙어 블록 주석이
// 거기서 닫히고, 뒤따르는 백틱이 문자열을 열어 스크립트 전체가 깨진다.
const SECS_PER_CALL = {qwen: 40, kormo: 18};

/* 브라우저가 base64로 부풀리므로(1.33배) 서버 상한 20MB에 닿기 전에 여기서 막는다.
   **서버 쪽 413은 본문을 다 받기 전에 연결을 끊어** 브라우저가 그 문구를 못 읽는다. */
const MAX_FILE = 14 * 1024 * 1024;

const RGN_KIND = {replace: '고쳤다', insert: '넣었다', delete: '뺐다'};

let REGIONS = [];
let modelKey = 'kormo-good';

/* 구간이 문서의 몇 번째 문단이었나. 넣기만 한 구간은 개정 전 쪽이 비므로 `—`다. */
const span = (at) => (at[1] < at[0] ? '—' : at[0] === at[1] ? `${at[0]}` : `${at[0]}–${at[1]}`);

const fileB64 = (file) => new Promise((ok, no) => {
  const reader = new FileReader();
  // data URL 로 읽으면 `data:...;base64,XXXX` 가 오므로 쉼표 뒤만 떼면 된다.
  reader.onload = () => ok(String(reader.result).slice(String(reader.result).indexOf(',') + 1));
  reader.onerror = () => no(reader.error);
  reader.readAsDataURL(file);
});

function checkFiles() {
  const files = [$('fileBefore').files[0], $('fileAfter').files[0]];
  const wrong = files.find((f) => f && !f.name.toLowerCase().endsWith('.hwpx'));
  const heavy = files.find((f) => f && f.size > MAX_FILE);
  let why = '';
  if (wrong) why = `${wrong.name} — hwpx가 아니다. 한글에서 hwpx로 저장해 올린다`;
  else if (heavy) why = `${heavy.name} — 너무 크다 (상한 ${MAX_FILE / 1048576 | 0}MB)`;
  else if (!files[0] || !files[1]) why = '두 파일을 다 고른다';
  $('scanNote').textContent = why;
  $('scan').disabled = why !== '';
}
['fileBefore', 'fileAfter'].forEach((id) =>
  $(id).addEventListener('change', checkFiles));

$('scan').addEventListener('click', async () => {
  $('scan').disabled = true;
  $('scanNote').textContent = '읽는 중…';
  try {
    const [before, after] = await Promise.all(
      [fileB64($('fileBefore').files[0]), fileB64($('fileAfter').files[0])]);
    const res = await fetch(API + '/extract', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({before, after}),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    REGIONS = data.regions;
    $('scanNote').textContent = '';
    $('regionHead').innerHTML = `문단 ${data.before_blocks} → ${data.after_blocks} · ` +
      `<b>바뀐 곳 ${REGIONS.length}군데</b>` +
      (REGIONS.length ? '' : ' — 두 문서가 같다');
    paintRegions();
    paintModels();
    $('regionBox').hidden = false;
  } catch (err) {
    $('scanNote').textContent = String(err.message || err);
    $('regionBox').hidden = true;
  }
  checkFiles();
});

function paintRegions() {
  $('regions').innerHTML = REGIONS.map((r, i) => `
    <div class='rgn${r.oversize ? ' off' : ''}'>
      <label class=hd><input type=checkbox data-rgn='${i}'${r.oversize ? '' : ' checked'}>
        ${i + 1}. ${RGN_KIND[r.kind] || r.kind} · 문단 ${span(r.before_at)} →
        ${span(r.after_at)} · ${r.chars}자${r.oversize ? ' · 너무 길어 기본 해제' : ''}</label>
      <div class=pv>${r.before_html || '<i>없음</i>'}</div>
      <div class=pv>${r.after_html || '<i>없음</i>'}</div>
    </div>`).join('');
  $('regions').querySelectorAll('[data-rgn]').forEach(
    (box) => box.addEventListener('change', paintRunNote));
  paintRunNote();
}

const pickedRegions = () => REGIONS.filter(
  (_, i) => $('regions').querySelector(`[data-rgn="${i}"]`).checked);

function paintModels() {
  if (!$('modelPick')) return;
  // 서버가 못 띄운 칸은 못 고르게 한다. 고르고 있던 것이 죽었으면 살아 있는 첫 칸으로.
  if (!ready.has(modelKey)) {
    const alive = CARDS.find((m) => ready.has(m.key));
    if (alive) modelKey = alive.key;
  }
  $('modelPick').innerHTML = CARDS.map((m) =>
    `<button class=chip data-model='${m.key}' aria-pressed='${m.key === modelKey}'` +
    `${ready.has(m.key) ? '' : ' disabled'}>${esc(m.label)}</button>`).join('');
  $('modelPick').querySelectorAll('[data-model]').forEach((b) =>
    b.addEventListener('click', () => {
      modelKey = b.dataset.model;
      paintModels();
      paintRunNote();
    }));
  paintRunNote();
}

function paintRunNote() {
  if (!$('runFile')) return;
  const n = REGIONS.length ? pickedRegions().length : 0;
  const per = modelKey.startsWith('qwen') ? SECS_PER_CALL.qwen : SECS_PER_CALL.kormo;
  const secs = n * per;
  $('runFile').disabled = n === 0 || !ready.has(modelKey);
  $('runNote').textContent = n === 0 ? '고른 곳이 없다'
    : `${n}군데 · 약 ${secs < 90 ? secs + '초' : Math.round(secs / 60) + '분'}`;
}

$('runFile').addEventListener('click', async () => {
  const chosen = pickedRegions();
  const meta = CARDS.find((m) => m.key === modelKey) || {label: modelKey};
  $('runFile').disabled = true;
  const states = chosen.map(() => ({pending: true}));
  const paint = () => {
    $('fileLive').innerHTML = chosen.map((r, i) => `<div class=mc>
      <div class=hd><span class=nm>${i + 1}. ${RGN_KIND[r.kind] || r.kind}</span>
        <span class=rn>${esc(meta.label)}</span></div>
      <div class=pv>${r.before_html || '<i>없음</i>'}</div>
      <div class=pv>${r.after_html || '<i>없음</i>'}</div>
      ${liveBody(states[i])}</div>`).join('');
  };
  paint();

  // 위 여섯 칸과 **같은 설정을 쓴다.** 규칙서를 떼거나 프리필을 바꾼 채로 물으면
  // 여기 답과 저기 답이 갈리는데, 두 칸이 한 화면에 있으므로 그러면 안 된다.
  const body = {
    model: modelKey, rules: $('useRules').checked,
    prefill: $('prefill').value, max_new_tokens: +$('maxTok').value,
  };
  // 하나씩 보낸다. 서버가 GPU마다 자물쇠를 걸어 어차피 줄을 서므로, 한꺼번에 던지면
  // 먼저 온 것이 먼저 그려지는 것만 잃는다.
  for (let i = 0; i < chosen.length; i += 1) {
    $('runNote').textContent = `${chosen.length}군데 중 ${i + 1}군째…`;
    try {
      const res = await fetch(API + '/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...body, before: chosen[i].before, after: chosen[i].after}),
      });
      const data = await res.json();
      states[i] = data.ok ? data : {error: data.error || '알 수 없는 오류'};
    } catch (err) {
      states[i] = {error: String(err)};
    }
    paint();
  }
  $('runFile').disabled = false;
  paintRunNote();
});

/* ---------------------------------------------------------------- 시작 */
const first = PICKS.length ? ITEMS.findIndex((i) => i.id === PICKS[0].id) : 0;
renderItem(first >= 0 ? first : 0);
drawAll('교사일치');
probe();
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    items = load_items()
    picks = curate(items)
    series = load_series()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page1(items, picks, series), encoding="utf-8")

    size = out.stat().st_size / 1024 / 1024
    print(f"조문 {len(items)}건 · 모델 {len(CATALOG)}칸 · 계열 {len(series)}개")
    for pick in picks:
        print(f"  고름: {pick['why']} -- {pick['id']}")
    print(f"\n저장: {out}  ({size:.1f}MB)")


if __name__ == "__main__":
    main()
