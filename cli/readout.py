"""홀드아웃 채점 결과를 사람이 눈으로 읽는 HTML 대조표로 만든다.

**판정만 보면 안 되는 것이 이 표의 존재 이유다.** `verdict`와 `교사일치`는 `judgement`
한 칸만 본다. 그런데 그 한 칸을 맞히면서 **`(대상, 방향)`을 다르게 읽은 건**이 따로 있고,
표에는 그것이 안 남는다(`docs/PLAN.md` 4절). 그래서 여기서는 판정과 라벨을 **따로 세고
따로 거른다.**

**라벨일치를 셀 때 빈 집합끼리 맞은 것을 빼는 것이 핵심이다.** negative 건은 교사도
모델도 `labels`가 비어 있어 자동으로 일치가 되는데, 그것까지 세면 `delora`가 40.5%로
보인다. 교사가 라벨을 단 건만 세면 **19.2%**다. `labels`가 비면 AM2·AM3이 검사할 것이
없어 자동 만점이 되는 것과 **똑같은 함정**이고, 여기서 또 밟지 않으려고 두 값을 다 적어
둔다.

**실험을 여럿 주면 탭으로 묶는다.** 잘한 것 하나만 보면 그 값이 높은 건지 낮은 건지
모른다 -- 같은 37건을 다른 모델이 어떻게 읽었는지 옆에 놓아야 차이가 보인다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.readout --run runs/delora
    python -m cli.readout --run runs/delora-r2 runs/delora runs/vera-r2 \
        --label '2차 최고' '1차 최고' '2차 최저' --out runs/판독표-r2.html
"""
from __future__ import annotations

import argparse
import html
import json
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = "data/20260811__annotate__v2.2/holdout.jsonl"


def read_jsonl(path: Path) -> dict[str, dict]:
    """`id`를 열쇠로 하는 사전으로 읽는다. 두 파일을 `id`로 맞붙이기 때문이다."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def label_set(labels) -> set[tuple[str, str]]:
    """`(대상, 방향)`만 남긴 집합. `근거`는 자유 문장이라 비교에 안 쓴다.

    어휘가 고정돼 있어(대상 7종 · 방향 5종) 집합 비교가 성립한다. 순서는 뜻이 없으므로
    집합으로 본다 -- 같은 라벨을 순서만 바꿔 낸 것을 틀렸다고 하면 안 된다.
    """
    out = set()
    for item in labels or []:
        if isinstance(item, dict):
            out.add((item.get("대상", ""), item.get("방향", "")))
        else:
            out.add(tuple(item))
    return out


def diff_html(before: str, after: str) -> tuple[str, str]:
    """바뀐 자리를 `<del>`·`<ins>`로 감싼 개정 전후 두 덩어리를 돌려준다.

    글자 단위로 본다. 낱말 단위로 자르면 조사 하나가 바뀐 것(`제출해야` -> `제출해`)이
    낱말 통째로 바뀐 것처럼 보여, 이 데이터에서 제일 흔한 미세 개정이 과장된다.
    """
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old, new = html.escape(before[i1:i2]), html.escape(after[j1:j2])
        if tag == "equal":
            left.append(old)
            right.append(new)
        else:
            if old:
                left.append(f"<del>{old}</del>")
            if new:
                right.append(f"<ins>{new}</ins>")
    return "".join(left), "".join(right)


def tags_html(mine: set, theirs: set, extra_class: str) -> str:
    """라벨 한 쪽을 칩으로 그린다. 상대에게 있으면 `hit`, 없으면 `extra_class`."""
    if not mine:
        return "<span class='tag none'>라벨 없음</span>"
    chips = []
    for 대상, 방향 in sorted(mine):
        kind = "hit" if (대상, 방향) in theirs else extra_class
        chips.append(f"<span class='tag {kind}'>{html.escape(대상)} · {html.escape(방향)}</span>")
    return "".join(chips)


def reading_html(row: dict, tokens: int | None) -> str:
    """한 쪽이 읽어낸 것(라벨 표 · 영향 목록 · 한 문장)을 그린다."""
    meta = f" <span class='meta'>{tokens}토큰</span>" if tokens is not None else ""
    who = "모델이 읽은 것" if tokens is not None else "교사가 읽은 것"
    parts = [f"<h4>{who}{meta}</h4>"]

    labels = row.get("labels") or []
    if labels:
        rows = "".join(
            f"<tr><td class='k'>{html.escape(str(l.get('대상', '')))}</td>"
            f"<td class='k'>{html.escape(str(l.get('방향', '')))}</td>"
            f"<td class='g'>{html.escape(str(l.get('근거', '')))}</td></tr>"
            for l in labels)
        parts.append("<table class='lab'><thead><tr><th>대상</th><th>방향</th>"
                     f"<th>근거</th></tr></thead><tbody>{rows}</tbody></table>")
    else:
        parts.append("<p class=\"none\">없음</p>")

    impacts = row.get("impacts") or []
    if impacts:
        items = "".join(
            f"<li><span class='who'>{html.escape(str(i.get('주체', '')))}</span>"
            f"<span class='what'>{html.escape(str(i.get('영향', '')))}</span></li>"
            for i in impacts)
        parts.append(f"<ul class='imp'>{items}</ul>")
    else:
        parts.append("<p class=\"none\">없음</p>")

    sentence = (row.get("direct_impact") or "").strip()
    parts.append(f"<p class=\"sent\">{html.escape(sentence)}</p>" if sentence
                 else "<p class=\"sent\"><span class=none>문장 없음</span></p>")
    return "".join(parts)


def build(run_dir: Path, holdout_path: Path, pid: int = 0, role: str = "",
          mode: str = "full") -> tuple[str, dict]:
    teacher = read_jsonl(holdout_path)
    model = read_jsonl(run_dir / "eval" / "records.jsonl")
    shared = [i for i in teacher if i in model]

    cases = []
    for rid in shared:
        t, m = teacher[rid], model[rid]
        t_labels, m_labels = label_set(t.get("labels")), label_set(m.get("labels"))
        cases.append({
            "id": rid, "teacher": t, "model": m,
            "t_labels": t_labels, "m_labels": m_labels,
            "jud_ok": t["judgement"] == m["judgement"],
            "lab_ok": t_labels == m_labels,
            # 교사가 라벨을 안 단 건은 라벨 정답률의 분모에서 뺀다. 빈 집합끼리 맞은
            # 것을 세면 negative가 많을수록 점수가 올라간다.
            "scored": bool(t_labels),
        })

    if mode == "sentences":
        # **`조용히 틀림`을 맨 위로 올린다.** 판정을 맞히고 라벨만 다른 건이 이 표가
        # 답하려는 질문이다 -- "라벨이 달라도 문장은 얼추 같은가". 판정부터 틀린 건은
        # 라벨을 볼 것도 없이 다른 이야기라 뒤로 보낸다.
        cases.sort(key=lambda c: (not (c["jud_ok"] and not c["lab_ok"]),
                                  c["jud_ok"], c["id"]))
    else:
        # 제일 나쁜 것을 맨 위로. 판정이 틀린 것 -> 라벨이 틀린 것 -> 맞은 것 순이다.
        cases.sort(key=lambda c: (c["jud_ok"], c["lab_ok"], c["id"]))

    total = len(cases)
    jud_bad = [c for c in cases if not c["jud_ok"]]
    lab_bad = [c for c in cases if not c["lab_ok"]]
    scored = [c for c in cases if c["scored"]]
    scored_ok = [c for c in scored if c["lab_ok"]]
    quiet = [c for c in cases if c["jud_ok"] and not c["lab_ok"]]
    stats = {
        "총": total,
        "판정일치": total - len(jud_bad),
        "판정불일치": len(jud_bad),
        "라벨불일치": len(lab_bad),
        "조용히틀림": len(quiet),
        "라벨분모": len(scored),
        "라벨일치": len(scored_ok),
    }

    articles = []
    for c in cases:
        t, m = c["teacher"], c["model"]
        classes = ["case"]
        if not c["jud_ok"]:
            classes += ["bad", "jud-bad"]
        if not c["lab_ok"]:
            classes.append("lab-bad")
        if not c["jud_ok"]:
            mark = "판정 불일치"
        elif not c["lab_ok"]:
            mark = "라벨 불일치"
        else:
            mark = "일치"

        left, right = diff_html(t.get("before", ""), t.get("after", ""))
        short = t.get("before_id", c["id"]).split("-")[-1]
        tokens = m.get("new_tokens")

        if mode == "sentences":
            # **문장만 앞으로 꺼낸다.** 이 표가 묻는 것은 "라벨이 달라도 문장은 얼추
            # 같은가" 하나뿐이라, 조문과 라벨은 접어 두고 필요할 때만 편다.
            articles.append(f"""
<article class="{' '.join(classes + (['quiet'] if c['jud_ok'] and not c['lab_ok'] else []))}">
  <header class="chead">
    <span class="cid">{html.escape(short)}</span>
    <span class="verdicts">
      <span class="lab-sm">교사</span><span class="pill {t['judgement']}">{t['judgement']}</span>
      <span class="arrow" aria-hidden="true">&rarr;</span>
      <span class="lab-sm">모델</span><span class="pill {m['judgement']}">{m['judgement']}</span>
    </span>
    <span class="mark">{mark}</span>
  </header>

  <div class="sents">
    <div class="srow m"><h4>모델</h4><p>{html.escape((m.get('direct_impact') or '').strip()) or '<span class=none>문장 없음</span>'}</p></div>
    <div class="srow t"><h4>교사</h4><p>{html.escape((t.get('direct_impact') or '').strip()) or '<span class=none>문장 없음</span>'}</p></div>
  </div>

  <details class="more">
    <summary>라벨과 개정 전후 보기</summary>
    <div class="labelbar">
      <div class="side"><span class="who">교사 라벨</span>{tags_html(c['t_labels'], c['m_labels'], 'miss')}</div>
      <div class="side"><span class="who">모델 라벨</span>{tags_html(c['m_labels'], c['t_labels'], 'extra')}</div>
    </div>
    <div class="table-wrap">
      <div class="daebi">
        <div class="col"><h4>개정 전</h4><p class="jomun">{left}</p></div>
        <div class="col"><h4>개정 후</h4><p class="jomun">{right}</p></div>
      </div>
    </div>
  </details>
</article>""")
            continue

        articles.append(f"""
<article class="{' '.join(classes)}">
  <header class="chead">
    <span class="cid">{html.escape(short)}</span>
    <span class="verdicts">
      <span class="lab-sm">교사</span><span class="pill {t['judgement']}">{t['judgement']}</span>
      <span class="arrow" aria-hidden="true">&rarr;</span>
      <span class="lab-sm">모델</span><span class="pill {m['judgement']}">{m['judgement']}</span>
    </span>
    <span class="mark">{mark}</span>
  </header>

  <div class="table-wrap">
    <div class="daebi">
      <div class="col"><h4>개정 전</h4><p class="jomun">{left}</p></div>
      <div class="col"><h4>개정 후</h4><p class="jomun">{right}</p></div>
    </div>
  </div>

  <div class="labelbar">
    <div class="side"><span class="who">교사 라벨</span>{tags_html(c['t_labels'], c['m_labels'], 'miss')}</div>
    <div class="side"><span class="who">모델 라벨</span>{tags_html(c['m_labels'], c['t_labels'], 'extra')}</div>
  </div>

  <div class="readings">
    <section class="reading model">{reading_html(m, tokens)}</section>
    <section class="reading teacher">{reading_html(t, None)}</section>
  </div>
</article>""")

    rate = stats["라벨일치"] / stats["라벨분모"] if stats["라벨분모"] else 0
    jud_rate = stats["판정일치"] / total if total else 0

    # **`sentence` 조건은 라벨을 아예 안 낸다.** 그러면 라벨일치가 구조적으로 0이 되는데,
    # 그것을 실력으로 읽으면 잘한 실험이 통째로 탈락한다 -- AM8s가 `sentence`에서
    # 자동 탈락을 만드는 것과 같은 자리다(`docs/TODO.md`). 숫자를 지우고 이유를 적는다.
    emits_labels = any(c["m_labels"] for c in cases)
    stats["라벨을냄"] = emits_labels
    if emits_labels:
        label_stat = (f"<dd>{rate:.1%}<small>{stats['라벨일치']}/{stats['라벨분모']}</small></dd>")
        label_note = (f"<p class=\"standfirst\"><strong>라벨일치의 분모는 "
                      f"{stats['라벨분모']}건이다</strong> &mdash; 교사가 라벨을 단 건만 센다."
                      f" 양쪽 다 비어서 저절로 맞은 {total - stats['라벨분모']}건을 넣으면"
                      f" 이 값이 부풀고, 그것은 <code>labels</code>가 비면"
                      f" AM2&middot;AM3이 자동 만점이 되는 것과 같은 함정이다.</p>")
    else:
        label_stat = "<dd>해당 없음<small>라벨을 안 내는 조건</small></dd>"
        label_note = ("<p class=\"standfirst\"><strong>이 실험은 라벨을 한 건도 내지"
                      " 않는다.</strong> <code>sentence</code> 조건이라 한 문장만 내고"
                      " <code>labels</code>를 안 쓴다 &mdash; <strong>라벨일치 0%는 실력이"
                      " 아니라 구조다.</strong> 이 표에서는 판정과 문장만 읽는다.</p>")

    quiet_button = (f'<button data-f="quiet" aria-pressed="false">'
                    f'조용히 틀림 {stats["조용히틀림"]}건</button>'
                    if mode == "sentences" else "")
    if mode == "sentences":
        # 문장 모드는 묻는 것이 하나뿐이라 머리말을 그것으로 바꾼다.
        lede = ("<strong>이 표가 묻는 것은 하나다 &mdash; 라벨이 달라도 "
                f"<code>direct_impact</code> 문장은 얼추 같은가.</strong> 그래서 "
                f"<strong>조용히 틀림 {stats['조용히틀림']}건</strong>(판정은 맞고 라벨만 "
                "다른 건)을 맨 위로 올렸다. 라벨과 개정 전후는 접어 두었다 &mdash; "
                "필요하면 각 건에서 편다.")
    else:
        lede = ("개정 전후 조문을 나란히 놓고 바뀐 자리를 표시한 뒤, 모델이 읽어낸 "
                "것과 교사가 읽어낸 것을 아래에 붙였다. " + (
                    f"<strong>판정을 맞히고도 라벨을 다르게 읽은 {stats['조용히틀림']}건이 "
                    "이 표의 요점이다</strong> &mdash; 판정만 세는 눈금에는 안 남는 값이다."
                    if emits_labels else
                    "<strong>판정과 문장만 읽는 표다</strong> &mdash; 이 조건은 라벨을 안 낸다."))
    panel = f"""<section class="panel{' on' if pid == 0 else ''}" id="p{pid}">
<header class="masthead">
  <p class="eyebrow">{html.escape(role)} &middot; {html.escape(run_dir.name)} &middot; 홀드아웃 {total}건</p>
  <h1>개정 해석 대조표</h1>
  <p class="standfirst">{lede}</p>
  <dl class="stats">
    <div class="stat"><dt>판정일치</dt><dd>{jud_rate:.1%}<small>{stats['판정일치']}/{total}</small></dd></div>
    <div class="stat"><dt>라벨일치</dt>{label_stat}</div>
    <div class="stat"><dt>판정 불일치</dt><dd>{stats['판정불일치']}<small>건</small></dd></div>
    <div class="stat"><dt>조용히 틀림</dt><dd>{stats['조용히틀림']}<small>판정만 맞음</small></dd></div>
  </dl>
  {label_note}
</header>

<div class="controls">
  <button data-f="all" aria-pressed="true">전체 {total}건</button>
  {quiet_button}
  <button data-f="only" aria-pressed="false">판정 불일치 {stats['판정불일치']}건</button>
  <button data-f="lab" aria-pressed="false">라벨 불일치 {stats['라벨불일치']}건</button>
  <span class="legend"><span><del>지운 자리</del></span><span><ins>넣은 자리</ins></span>
    <span><span class="tag miss">교사만</span></span><span><span class="tag extra">모델만</span></span></span>
</div>
{''.join(articles)}
</section>"""
    return panel, stats


def page(panels: list[str], tabs: list[tuple[str, str]]) -> str:
    """판 여럿을 탭 하나로 묶는다.

    **필터는 판마다 따로 건다.** 예전에는 `body`에 `only-bad`를 걸었는데, 판이 여럿이면
    안 보이는 판까지 같이 걸러져 탭을 옮겼을 때 왜 비었는지 알 수 없게 된다.
    """
    css = (ROOT / "cli" / "readout.css").read_text(encoding="utf-8")
    bar = "".join(
        f'<button class="tab" role="tab" data-for="p{i}" '
        f'aria-selected="{str(i == 0).lower()}">'
        f'<span class="trole">{html.escape(role)}</span>'
        f'<span class="tname">{html.escape(name)}</span></button>'
        for i, (role, name) in enumerate(tabs))
    named = (html.escape(tabs[0][1]) if len(tabs) == 1
             else f"{html.escape(tabs[0][1])} 외 {len(tabs) - 1}")
    return f"""<title>개정 해석 대조표 &middot; {named}</title>
<style>{css}</style>
<div class="wrap">
<nav class="tabs" role="tablist">{bar}</nav>
{''.join(panels)}
</div>
<script>
for (const tab of document.querySelectorAll('.tab')) {{
  tab.addEventListener('click', () => {{
    for (const other of document.querySelectorAll('.tab'))
      other.setAttribute('aria-selected', String(other === tab));
    for (const panel of document.querySelectorAll('.panel'))
      panel.classList.toggle('on', panel.id === tab.dataset.for);
    window.scrollTo({{top: 0}});
  }});
}}
for (const button of document.querySelectorAll('.controls button')) {{
  button.addEventListener('click', () => {{
    const panel = button.closest('.panel');
    panel.classList.toggle('only-bad', button.dataset.f === 'only');
    panel.classList.toggle('only-lab', button.dataset.f === 'lab');
    panel.classList.toggle('only-quiet', button.dataset.f === 'quiet');
    for (const other of panel.querySelectorAll('.controls button'))
      other.setAttribute('aria-pressed', String(other === button));
  }});
}}
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, nargs="+", type=Path,
                    help="실험 폴더. 여럿 주면 탭으로 묶는다. 그 안의 eval/records.jsonl을 읽는다")
    ap.add_argument("--label", nargs="+", default=None,
                    help="탭에 적을 역할. --run 과 개수가 같아야 한다 (예: '2차 최고')")
    ap.add_argument("--sentences", action="store_true",
                    help="direct_impact 문장만 앞으로 꺼낸 판. 조용히 틀림이 맨 위로 온다")
    ap.add_argument("--data", default=HOLDOUT, help="교사 정답이 든 홀드아웃")
    ap.add_argument("--out", type=Path, default=None,
                    help="안 주면 runs/판독표.html")
    args = ap.parse_args()

    labels = args.label or [""] * len(args.run)
    if len(labels) != len(args.run):
        ap.error(f"--label {len(labels)}개가 --run {len(args.run)}개와 안 맞는다")

    panels, tabs, every = [], [], []
    for pid, (run, role) in enumerate(zip(args.run, labels)):
        run_dir = run if run.is_absolute() else ROOT / run
        panel, stats = build(run_dir, ROOT / args.data, pid, role,
                             mode="sentences" if args.sentences else "full")
        panels.append(panel)
        tabs.append((role or f"판 {pid + 1}", run_dir.name))
        every.append((run_dir.name, role, stats))

    out = args.out or (ROOT / "runs" / "판독표.html")
    out.write_text(page(panels, tabs), encoding="utf-8")

    for name, role, stats in every:
        print(f"{role or '판'} · {name} -- 홀드아웃 {stats['총']}건")
        print(f"  판정일치      {stats['판정일치']}/{stats['총']}")
        print(f"  판정 불일치   {stats['판정불일치']}")
        if stats["라벨을냄"]:
            print(f"  라벨일치      {stats['라벨일치']}/{stats['라벨분모']}"
                  f"   (교사가 라벨을 단 건만 센다)")
            print(f"  조용히 틀림   {stats['조용히틀림']}   판정은 맞고 라벨이 다르다")
        else:
            print("  라벨일치      해당 없음")
            print("    ! 이 실험은 라벨을 한 건도 내지 않습니다(sentence 조건).")
            print("      라벨일치 0%는 실력이 아니라 구조이므로 다른 실험과 나란히 놓지 마십시오.")
        print()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
