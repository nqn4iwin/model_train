"""실습 페이지를 만든다. **설명 없는 도구 한 장이다.**

개정 전후 문서를 넣으면 고른 모델들이 각자 판정과 해석을 낸다. 그게 전부다.
읽을거리(저장된 답 비교·학습 곡선)는 여기 없다 -- 그쪽은 보고서 몫이다.

**`runs/`를 안 읽는다.** 홀드아웃 원문만 있으면 지어진다. 그래서 채점 결과가 어디
있든(서버든 로컬이든) 상관없이 만들어진다.

**모델 목록은 `cli/serve.py`에서 가져온다.** 페이지와 서버가 다른 모델을 가리키면
고른 것과 답한 것이 어긋나므로 목록은 한 군데에만 둔다. 화면에 붙는 이름(A·B·C…)도
저쪽 `letter`다.

**G(GPT)만 여기서 더한다.** 서버에 그 모델이 없기 때문이다 -- 칸만 세워 두었고 API는
나중에 붙인다. 그때까지 그 칸은 「API 연결 안 됨」으로 답한다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.pages
    python -m cli.pages --out visualizations/실습.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli.serve import CATALOG, DEFAULT_PORT, MIN_INPUT_CHARS, PAIR_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]

HOLDOUT = ROOT / "data/20260821__annotate__v2.2-run2A/holdout.jsonl"
OUT = ROOT / "visualizations/정리_1_결과.html"

# 처음부터 켜져 있는 칸. **학습한 것 하나와 안 한 것 하나**를 켜 두어, 누르자마자
# 둘의 차이가 한 화면에 보이게 한다.
DEFAULT_ON = ["kormo-good", "gpt-nolearn"]


def load_items() -> list[dict]:
    """홀드아웃에서 **positive만** 골라 온다.

    「테스트 데이터에서 가져오기」는 모델이 무엇을 하는지 보여 주는 단추다. 안 바뀐
    조문(negative)을 물으면 「바뀐 것 없음」이 나와 볼 것이 없으므로 뺀다.

    **`MIN_INPUT_CHARS`보다 짧은 것도 뺀다.** 안 빼면 불러온 것을 서버가 「너무
    짧습니다」로 되돌려주는 꼴이 된다 -- 자기 단추가 준 것을 자기가 막는 셈이다.
    113건 중 8건이 여기서 빠져 105건이 남는다.
    """
    items = []
    for line in HOLDOUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("judgement") != "positive":
            continue
        if min(len(row["before"]), len(row["after"])) < MIN_INPUT_CHARS:
            continue
        items.append({"id": row["id"], "before": row["before"],
                      "after": row["after"]})
    return items


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
html{font-size:20px}
body{margin:0;padding:2.4rem 1.5rem 8rem;background:var(--bg);color:var(--fg);
  font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;
  font-size:1rem;line-height:1.7;-webkit-font-smoothing:antialiased}
main{max-width:74rem;margin:0 auto}

h1{font-size:2.4rem;font-weight:700;letter-spacing:-.02em;margin:0 0 .4rem}
h2{font-size:1.15rem;font-weight:650;color:var(--fg2);margin:2.2rem 0 .7rem}

.status{display:flex;gap:.6rem;align-items:center;font-size:.85rem;
  color:var(--mut);margin:0 0 2rem}
.dot{width:.55rem;height:.55rem;border-radius:50%;background:var(--mut);flex:none}
.dot.on{background:var(--ins)}
.dot.off{background:var(--del)}

button{font:inherit;cursor:pointer}
button.load{font-size:1.05rem;font-weight:600;padding:.7rem 1.5rem;
  border-radius:10px;border:2px solid var(--line2);background:var(--card);
  color:var(--fg)}
button.load:hover{border-color:var(--fg);background:var(--card2)}
.hint{color:var(--mut);font-size:.85rem;margin-left:.8rem}

.io{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin:.6rem 0}
@media (max-width:900px){.io{grid-template-columns:1fr}}
.io > div{display:flex;flex-direction:column;gap:.55rem}
.io h2{margin:0}
input[type=file]{font:inherit;font-size:.85rem;color:var(--fg2);padding:.55rem;
  border:2px dashed var(--line2);border-radius:10px;background:var(--card);
  width:100%}
textarea{font:inherit;font-size:1.05rem;line-height:1.65;width:100%;
  min-height:11rem;padding:.9rem 1rem;resize:vertical;border-radius:10px;
  border:2px solid var(--line);background:var(--card);color:var(--fg)}
textarea:focus{outline:none;border-color:var(--accent)}
textarea:disabled{opacity:.4}

.notice{font-size:1.05rem;font-weight:600;padding:.9rem 1.2rem;border-radius:10px;
  margin:1.2rem 0;display:none}
.notice.bad{display:block;background:var(--delb);color:var(--del)}
.notice.ok{display:block;background:var(--card);color:var(--fg2);font-weight:500}

.rgn{border:2px solid var(--line);border-radius:10px;background:var(--card);
  padding:.8rem 1rem;margin:.5rem 0;cursor:pointer;display:block}
.rgn:hover{border-color:var(--line2)}
.rgn.on{border-color:var(--accent);background:var(--card2)}
.rgn .pv{display:block;font-size:.95rem;line-height:1.6;overflow-wrap:anywhere;
  margin-top:.35rem}
del{background:var(--delb);color:var(--del);text-decoration:none;
  border-radius:3px;padding:0 .12em}
ins{background:var(--insb);color:var(--ins);text-decoration:none;
  border-radius:3px;padding:0 .12em}

.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(23rem,1fr));
  gap:.5rem;margin:.6rem 0}
label.mo{display:flex;gap:.7rem;align-items:center;padding:.75rem 1rem;
  border:2px solid var(--line);border-radius:10px;background:var(--card);
  font-size:1.02rem;cursor:pointer}
label.mo:hover{border-color:var(--line2)}
label.mo.on{border-color:var(--accent)}
label.mo.dead{opacity:.4;cursor:not-allowed}
label.mo input{width:1.15rem;height:1.15rem;accent-color:var(--accent);flex:none}
label.mo b{font-family:var(--mono);font-size:1.1rem;color:var(--accent);flex:none}

button.run{display:block;width:100%;margin:2.2rem 0 0;padding:1.5rem;
  font-size:1.8rem;font-weight:750;letter-spacing:-.02em;border-radius:14px;
  border:none;background:var(--fg);color:var(--bg)}
button.run:disabled{opacity:.28;cursor:not-allowed}

.spin{display:none;margin:2.5rem auto;width:3.2rem;height:3.2rem;
  border:5px solid var(--line);border-top-color:var(--accent);border-radius:50%;
  animation:sp .8s linear infinite}
.spin.on{display:block}
@keyframes sp{to{transform:rotate(360deg)}}

.out{display:flex;flex-direction:column;gap:1rem;margin:2rem 0 0}
.rc{border:2px solid var(--line);border-radius:12px;background:var(--card);
  padding:1.3rem 1.5rem;display:flex;flex-direction:column;gap:.7rem}
.rc .row{display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
.rc .k{font-size:1rem;color:var(--mut);flex:none;min-width:9.5rem}
.rc .v{font-size:1.15rem;font-weight:600}
.rc .ans{font-size:1.25rem;line-height:1.75;font-weight:500;flex:1}
.badge{display:inline-block;font-size:1.05rem;font-weight:700;padding:.15rem .7rem;
  border-radius:7px;font-family:var(--mono)}
.badge.positive{color:var(--pos);background:var(--posb)}
.badge.negative{color:var(--neg);background:var(--negb)}
.badge.none{color:var(--none);background:var(--noneb)}
.rc.wait .v,.rc.wait .ans{color:var(--mut);font-weight:400}
.rc.err{border-color:var(--del)}
.rc.err .ans{color:var(--del)}
"""

JS = r"""
const CARDS = JSON.parse(document.getElementById('CARDS').textContent);
const ITEMS = JSON.parse(document.getElementById('ITEMS').textContent);
const CONF  = JSON.parse(document.getElementById('CONF').textContent);

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => (
  {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

const API = location.protocol.startsWith('http')
  ? location.origin : 'http://localhost:__PORT__';

let ready = new Set();     // 서버가 실제로 띄운 모델
let regions = [];          // 맞댄 결과로 나온 바뀐 구간
let picked = 0;            // 그중 고른 것
let holdoutAt = -1;        // 「홀드아웃 불러쓰기」가 어디까지 갔나

/* --------------------------------------------------------- 서버 상태 */

async function probe() {
  try {
    const res = await fetch(API + '/models', {cache: 'no-store'});
    const data = await res.json();
    ready = new Set(data.models.filter((m) => m.ready).map((m) => m.key));
    $('dot').className = ready.size ? 'dot on' : 'dot off';
    $('stat').textContent = ready.size + '/' + data.models.length;
  } catch (err) {
    ready = new Set();
    $('dot').className = 'dot off';
    $('stat').textContent = '서버 없음';
  }
  paintModels();
}

/* ----------------------------------------------------------- 모델 칸 */

function paintModels() {
  $('models').innerHTML = CARDS.map((m) => {
    const dead = !ready.has(m.key);
    const on = CONF.on.includes(m.key) && !dead;
    return "<label class='mo" + (on ? ' on' : '') + (dead ? ' dead' : '') + "'>" +
      "<input type=checkbox data-model='" + m.key + "'" + (on ? ' checked' : '') +
      (dead ? ' disabled' : '') + "><b>" + m.letter + "</b>" +
      '<span>' + esc(m.label) + '</span></label>';
  }).join('');
  $('models').querySelectorAll('input[data-model]').forEach((box) =>
    box.addEventListener('change', () => {
      box.closest('label').classList.toggle('on', box.checked);
      refresh();
    }));
  refresh();
}

const chosenModels = () => CARDS.filter((m) => {
  const box = $('models').querySelector('[data-model="' + m.key + '"]');
  return box && box.checked;
});

/* --------------------------------------------------------- 넣는 자리 */

const fileOf = (side) => $(side === 'b' ? 'fb' : 'fa').files[0] || null;
const textOf = (side) => $(side === 'b' ? 'tb' : 'ta').value.trim();

const readB64 = (file) => new Promise((ok, no) => {
  const reader = new FileReader();
  reader.onload = () => ok(String(reader.result).slice(
    String(reader.result).indexOf(',') + 1));
  reader.onerror = () => no(reader.error);
  reader.readAsDataURL(file);
});

function say(text, bad) {
  $('notice').className = 'notice' + (text ? (bad ? ' bad' : ' ok') : '');
  $('notice').textContent = text || '';
}

function refresh() {
  $('run').disabled = regions.length === 0 || chosenModels().length === 0;
}

/* 두 칸이 다 차면 서버에 맞대 본다. 파일을 고른 쪽의 글칸은 잠근다. */
let pending = null;
async function sync() {
  ['b', 'a'].forEach((side) => {
    const box = $(side === 'b' ? 'tb' : 'ta');
    box.disabled = !!fileOf(side);
    if (box.disabled) box.value = '';
  });

  regions = [];
  picked = 0;
  $('regions').innerHTML = '';
  $('out').innerHTML = '';

  const files = [fileOf('b'), fileOf('a')];
  const texts = [textOf('b'), textOf('a')];
  const wrong = files.find((f) => f && !f.name.toLowerCase().endsWith('.hwpx'));
  if (wrong) {
    say('지금은 hwpx 파일만 지원합니다. 변환해주시기 바랍니다', true);
    refresh();
    return;
  }
  // 한쪽만 파일이면 막는다. 파일에서는 바뀐 구간을 골라야 하는데 다른 쪽이 글이면
  // 무엇과 무엇을 맞댈지가 정해지지 않는다.
  if (files[0] && !files[1]) { say('개정 후도 hwpx 파일로 넣어 주세요'); refresh(); return; }
  if (files[1] && !files[0]) { say('개정 전도 hwpx 파일로 넣어 주세요'); refresh(); return; }
  if (!files[0] && !(texts[0] && texts[1])) { say(''); refresh(); return; }

  say('맞춰 보는 중…');
  const token = {};
  pending = token;
  let body;
  if (files[0]) {
    const pair = await Promise.all(files.map(readB64));
    body = {before: pair[0], after: pair[1]};
  } else {
    body = {before_text: texts[0], after_text: texts[1]};
  }
  let data;
  try {
    const res = await fetch(API + '/extract', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    data = await res.json();
  } catch (err) {
    if (pending === token) { say('서버에 닿지 못했습니다', true); refresh(); }
    return;
  }
  if (pending !== token) return;     // 그 사이에 입력이 또 바뀌었다

  if (!data.ok) { say(data.error, true); refresh(); return; }
  if (data.too_different) { say('두 문서가 너무 다릅니다', true); refresh(); return; }

  regions = data.regions;
  if (!regions.length) { say('바뀐 곳이 없습니다', true); refresh(); return; }
  if (regions.length > 1) { say('바뀐 곳을 하나 고르세요'); paintRegions(); }
  else { say(''); }
  refresh();
}

function paintRegions() {
  $('regions').innerHTML = regions.map((r, i) =>
    "<label class='rgn" + (i === picked ? ' on' : '') + "'>" +
    "<input type=radio name=rgn value='" + i + "'" +
    (i === picked ? ' checked' : '') + '>' +
    "<span class=pv>" + (r.before_html || '<i>없음</i>') + '</span>' +
    "<span class=pv>" + (r.after_html || '<i>없음</i>') + '</span></label>').join('');
  $('regions').querySelectorAll('input[name=rgn]').forEach((radio) =>
    radio.addEventListener('change', () => {
      picked = +radio.value;
      $('regions').querySelectorAll('.rgn').forEach((el, i) =>
        el.classList.toggle('on', i === picked));
    }));
}

['fb', 'fa'].forEach((id) => $(id).addEventListener('change', sync));
['tb', 'ta'].forEach((id) => {
  let timer;
  $(id).addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(sync, 400);
  });
});

/* --------------------------------------------------- 홀드아웃 불러쓰기 */

$('load').addEventListener('click', () => {
  if (!ITEMS.length) return;
  holdoutAt = (holdoutAt + 1) % ITEMS.length;
  const item = ITEMS[holdoutAt];
  ['fb', 'fa'].forEach((id) => { $(id).value = ''; });
  $('tb').disabled = false;
  $('ta').disabled = false;
  $('tb').value = item.before;
  $('ta').value = item.after;
  sync();
});

/* ------------------------------------------------------------- 분석 */

function resultCard(meta, state) {
  const judge = String((state.parsed || {}).judgement || '').trim();
  const badge = (judge === 'positive' || judge === 'negative')
    ? "<span class='badge " + judge + "'>" + judge + '</span>'
    : "<span class='badge none'>" + (state.pending ? '…' : '못 읽음') + '</span>';
  const answer = state.error ? state.error
    : state.pending ? '분석 중…'
    : (String((state.parsed || {}).direct_impact || '').trim() || '문장 없음');
  const cls = 'rc' + (state.pending ? ' wait' : '') + (state.error ? ' err' : '');
  return "<div class='" + cls + "'>" +
    '<div class=row><span class=k>선택모델</span><span class=v>' +
      meta.letter + ' | ' + esc(meta.label) + '</span></div>' +
    '<div class=row><span class=k>문서 의미 차이</span><span class=v>' +
      (state.error ? '—' : badge) + '</span></div>' +
    '<div class=row><span class=k>최종 응답</span><span class=ans>' +
      esc(answer) + '</span></div></div>';
}

$('run').addEventListener('click', async () => {
  const targets = chosenModels();
  const region = regions[picked];
  if (!region || !targets.length) return;

  $('run').disabled = true;
  $('spin').className = 'spin on';
  const states = targets.map(() => ({pending: true}));
  const paint = () => {
    $('out').innerHTML = targets.map((m, i) => resultCard(m, states[i])).join('');
  };
  paint();

  // 하나씩 보낸다. 서버가 GPU마다 자물쇠를 걸어 어차피 줄을 서므로, 한꺼번에 던지면
  // 먼저 온 것이 먼저 그려지는 것만 잃는다.
  for (let i = 0; i < targets.length; i += 1) {
    try {
      const res = await fetch(API + '/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: targets[i].key,
                              before: region.before, after: region.after}),
      });
      const data = await res.json();
      states[i] = data.ok ? data : {error: data.error || '알 수 없는 오류'};
    } catch (err) {
      states[i] = {error: String(err)};
    }
    paint();
  }
  $('spin').className = 'spin';
  $('run').disabled = false;
});

probe();
"""


def data_island(name: str, payload) -> str:
    """JSON을 스크립트 칸에 넣는다. `</`를 막는 것은 **본문 안의 `</script>`가 태그로
    읽혀 페이지가 거기서 끊기는 것**을 피하려는 것이다."""
    text = json.dumps(payload, ensure_ascii=False).replace("</", r"<\/")
    return f"<script type='application/json' id='{name}'>{text}</script>"


def build_page(items: list[dict]) -> str:
    cards = [{"key": e["key"], "letter": e["letter"], "label": e["label"]}
             for e in CATALOG]
    conf = {"on": DEFAULT_ON, "threshold": PAIR_THRESHOLD}

    return f"""<!doctype html>
<html lang=ko><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>문서 차이 해석 SLM</title>
<style>{CSS}</style>
<main>
<h1>문서 차이 해석 SLM</h1>
<div class=status><span class=dot id=dot></span><span id=stat>서버 확인 중</span></div>

<p><button class=load id=load>테스트 데이터에서 가져오기</button>
<span class=hint>다른 데이터를 보고 싶으면 버튼을 여러 번 클릭하세요.</span></p>

<div class=io>
<div><h2>개정 전</h2>
<input type=file id=fb accept=".hwpx">
<textarea id=tb placeholder="또는 조문을 붙여 넣으세요"></textarea></div>
<div><h2>개정 후</h2>
<input type=file id=fa accept=".hwpx">
<textarea id=ta placeholder="또는 조문을 붙여 넣으세요"></textarea></div>
</div>

<div class=notice id=notice></div>
<div id=regions></div>

<h2>모델</h2>
<div class=models id=models></div>

<button class=run id=run disabled>분석하기</button>
<div class=spin id=spin></div>
<div class=out id=out></div>
</main>
{data_island('CARDS', cards)}
{data_island('ITEMS', items)}
{data_island('CONF', conf)}
<script>{JS.replace('__PORT__', str(DEFAULT_PORT))}</script>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    items = load_items()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(items), encoding="utf-8")

    print(f"홀드아웃 positive {len(items)}건 · 모델 {len(CATALOG)}칸")
    print(f"저장: {out}  ({out.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
