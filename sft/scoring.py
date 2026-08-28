"""학생 모델(KORMo) 출력을 정답키 없이 채점한다. `data_collect`에서 옮겨 심은 것이다.

**왜 옮겨 심었나.** 원본은 `data_collect/training_data/interpret/`의 `annotate.py`와
`run.py`에 있는데, 두 파일 다 맨 위에서 `solar`를 import 한다. `solar.py`는 교사 모델
API를 부르는 모듈이라 API 키를 요구하고, 학습 서버에는 그 레포 자체가 없다. 채점에
실제로 쓰이는 것은 순수 함수 몇 개뿐이므로 그것만 가져왔다.

**2026-08-27 까지는 한 글자도 안 바꿨다. 그날 `parse_output` 하나가 갈라졌다.**
채점 로직이 갈라지면 교사 값과 학생 값을 한 표에 못 놓는다. `rubric.md`가 적어둔
사고 -- 이름이 같은데 잣대가 다르면 언젠가 누군가 반드시 한 줄에 놓는다 -- 가 여기서도
그대로 성립하므로, **갈라진 자리를 여기 적어 둔다.**

    parse_output   깨진 출력을 고쳐 읽는다. 원본은 못 읽고 0점을 준다
                   135건 자에서 84.3% -> 95.5%. 사유와 실측은 그 함수의 설명에 있다

**그러므로 `test_scoring.py`의 원본 대조는 이제 AM1 에서 어긋난다.** 어긋남이 전부
"원본 0점 -> 옮긴 것 1점" 방향인지 확인하고 넘어가는 것이지, 어긋남이 없어야 하는
것이 아니다. 나머지 함수는 그대로이므로 대조를 버리지 않는다. **`data_collect` 쪽에
이 파서를 옮길지는 아직 안 정했다**(`docs/TODO.md`).

가져온 곳:
    run.py       TARGETS · DIRECTIONS · parse_output · label_pairs · impact_subjects
                 RESTATEMENT_THRESHOLD
    annotate.py  score_blind · restatement_ratio

**AM4·AM5·AM7은 여기에 없다.** 사람이 붙인 정답키가 있어야 매겨지는데 원천 697건에는
없다. 채점 정의는 `data_collect/training_data/interpret/rubric.md`에 있다.
"""
from __future__ import annotations

import difflib
import json
import re

# rubric.md AM2가 쓰는 어휘 목록. 이 밖의 말이 하나라도 나오면 0점이다.
TARGETS = ["기한·시점", "수치·기준", "적용 범위", "수행 주체",
           "절차·요건", "제출물·기재사항", "명칭"]
DIRECTIONS = ["늘었다", "줄었다", "다른 값", "새로 생겼다", "없어졌다"]

# 라벨을 **어느 칸으로 볼지**. `label_match`가 이걸로 갈라 본다.
#
# 2026-08-24에 더했다. 그전에는 `(대상, 방향)` 쌍 하나뿐이라, 라벨일치가 낮을 때
# **대상을 못 고른 것인지 방향을 못 고른 것인지 알 수 없었다.** 갈라 보니 2차run에서
# 대상 42.4% · 방향 46.0%로 둘이 비슷하게 어려웠다 -- 한쪽이 범인일 거라는 짐작이
# 틀렸다. 쌍은 31.1%인데, 두 값을 곱한 19.5%보다 높으므로 **대상을 맞히면 방향도
# 맞히는 경향**이 있다.
#
# **`쌍`이 기본이고 그것이 기존 `라벨일치`와 같은 값이다.** 새 칸을 더해도 옛 표의
# 숫자가 안 움직이도록 기본값을 바꾸지 않는다.
PARTS = {
    "쌍": lambda pair: pair,
    "대상": lambda pair: pair[0],
    "방향": lambda pair: pair[1],
}

# 채점 항목 이름. `s`는 self-consistency로, 정답키 대신 모델 자기 판정으로 분기한다는
# 뜻이다. 평가 세트의 AM6·AM8과 잣대가 다르므로 나란히 놓지 않는다(rubric.md).
KEYS = ("AM1", "AM2", "AM3", "AM6s", "AM8s")

# AH1(재진술 아님)을 사람이 읽기 전에 명백한 복사를 걸러내는 값. 합격 판정에는 쓰지
# 않는다 -- 유사도가 낮은 실패가 실제로 있다. 짧은 블록에서 오탐이 나는 것도 확인돼
# 있으므로(rubric.md), 이 값으로 점수를 매기지 말고 분포만 본다.
RESTATEMENT_THRESHOLD = 0.60


_DECODER = json.JSONDecoder()

# 모델이 **문자열 구분자 자리에** 쓴 둥근 따옴표. 문장 안에 든 둥근 따옴표는 JSON 에
# 아무 문제가 없으므로 건드리면 안 된다 -- 그래서 다른 수를 다 쓴 뒤 마지막에 한 번만
# 갈아 보고, 그래도 안 읽히면 포기한다. 실측 2건짜리 자리다.
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"'}


def _strip_fence(text: str) -> str:
    """코드펜스(```)를 벗긴다. 옛 파서에서 그대로 가져온 부분이다."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    return cleaned


def _balanced(text: str) -> str | None:
    """첫 `{`부터 읽어 **괄호가 맞는 토막**을 낸다. 못 만들면 None.

    두 가지 일을 한다.

        온전히 닫혔으면   그 자리에서 끊는다 -- 뒤에 무엇이 붙어 있든 안 본다
        닫히다 말았으면   쓰다 만 꼬리를 지우고 열어 둔 것을 **역순으로** 닫는다

    **역순이 핵심이다.** `{ "labels": [ {` 까지 쓰다 끊긴 출력에 `}` 만 붙이면 여전히
    깨진다 -- 배열을 `]` 로 닫아야 한다. 2026-08-27 에 `}` 만 붙여 재보고 587건이
    안 살아난 것이 이 자리였다.
    """
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]
    stack: list[str] = []
    in_string = escaped = False
    for position, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
            if not stack:
                return text[:position + 1]
    # 여기까지 왔으면 닫히지 않은 것이다. 모델이 쓰다 만 자리를 정리한다.
    patched = text.rstrip()
    if in_string:
        patched += '"'
    # 값 없이 칸 이름만 쓰다 끊긴 꼬리(`, "대상": `)를 통째로 지운다.
    patched = re.sub(r',\s*"[^"]*"\s*:\s*$', "", patched)
    patched = re.sub(r"[,:]\s*$", "", patched)
    for opener in reversed(stack):
        patched += "}" if opener == "{" else "]"
    return patched


def parse_output(text: str) -> dict | None:
    """모델 출력에서 JSON 객체 하나를 꺼낸다. 코드펜스는 벗기고, **깨진 것은 고쳐 읽는다.**

    **2026-08-27 에 갈아 끼웠다. 그전에는 첫 `{` 부터 마지막 `}` 까지를 통째로 잘라
    한 번에 읽었다.** 그 방식이 지는 자리가 셋이었다.

        이중 객체     `{...}{...}` 를 하나로 잘라 읽으려다 깨진다
        미완결       `{ "labels": [ {` 까지 쓰다 끊긴 것을 아예 못 읽는다
        둥근 따옴표   구분자 자리에 `\u201d` 가 오면 깨진다

    **이중 객체는 모델 잘못이라기 어렵다.** 학습 정답(`sft.formatting.build_completion`)
    은 한 줄짜리 압축 JSON 인데 `cli/evaluate.py` 의 프리필은 여러 줄 들여쓴 모양으로
    출발시킨다. 그래서 모델이 **여러 줄 모양으로 한 번 닫고, 학습대로 한 줄 모양을 또
    낸다.** 앞의 것은 흠 없이 완결된 답이다.

    135건 자 실측(14,985건 · 실험 111개):

        옛 파서 12,635건 (84.3%)  ->  새 파서 14,304건 (95.5%)

        깨진 2,350건의 내역   이중 객체 243 · 닫히다 만 것 1,424 · 둥근 따옴표 2
                             못 읽음 681 (대부분 학습 전 기준선이 예시를 되풀이한 것)
        건져 낸 칸           판정 1,669 · labels 1,216 · **direct_impact 문장 1,324**

    **회귀는 0건이다.** 옛 파서가 읽던 12,635건이 하나도 안 깨졌고, 읽은 값이 달라진
    것도 없다. 새 파서는 옛 파서가 지던 자리에서만 이긴다.

    ── 이 교체가 끊는 것 ──────────────────────────────────────────────
    **AM1 이 곧 "이 함수가 읽어 내는가"이므로 2026-08-27 이전 표의 AM 값과 나란히
    놓으면 안 된다.** 표 전체를 `cli.rescore --write` 로 한꺼번에 다시 매겨 표 안에서는
    비교가 되게 했지만, 문서에 박혀 있는 옛 숫자는 옛 파서의 값이다.

    **`data_collect` 원본과 갈라졌다.** 이 모듈 머리말의 "한 글자도 바꾸지 않았다" 는
    이 함수에 대해서는 더 이상 참이 아니다. `tests/test_scoring.py` 의 원본 대조는
    이제 AM1 에서 어긋난다 -- 새 파서가 원본이 못 읽던 것을 읽기 때문이다.

    **붕괴 판정도 같이 움직인다.** `collapsed()` 는 읽힌 판정만 보는데, 못 읽은 26건
    안에 negative 24건이 들어 있던 줄이 있었다(`delora-sentence-s44-run2A`). 반대로
    negative 134건에 positive 1건이 건져지면서 **진짜 붕괴인데 붕괴 표시가 꺼지는 줄이
    넷** 생긴다. 그쪽은 `쏠림` 열이 99.3% 로 남으므로 그 열로 읽는다.
    """
    cleaned = _strip_fence(text)
    start = cleaned.find("{")
    if start < 0:
        return None
    # 1. 앞에서부터 객체 하나만 떼어 읽는다. 뒤에 무엇이 붙어 있어도 상관없다.
    try:
        result, _ = _DECODER.raw_decode(cleaned[start:])
        if isinstance(result, dict):
            return result
    except ValueError:
        pass
    # 2. 안 되면 괄호를 맞춰 고쳐 읽는다. 둥근 따옴표 치환은 맨 마지막이다.
    for candidate in (cleaned, "".join(_SMART_QUOTES.get(c, c) for c in cleaned)):
        patched = _balanced(candidate)
        if patched is None:
            continue
        try:
            result = json.loads(patched)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None


def label_pairs(labels) -> list[tuple[str, str]] | None:
    """labels 배열을 (대상, 방향) 튜플 목록으로 바꾼다. 모양이 어긋나면 None."""
    if not isinstance(labels, list):
        return None
    pairs = []
    for entry in labels:
        if not isinstance(entry, dict):
            return None
        pairs.append((str(entry.get("대상", "")), str(entry.get("방향", ""))))
    return pairs


def label_match(model_labels, teacher_labels, part: str = "쌍") -> bool | None:
    """모델이 읽은 `(대상, 방향)` 집합이 교사와 같은가. **교사가 라벨을 안 달았으면 None.**

    None은 "틀렸다"가 아니라 **"이 건은 분모에서 뺀다"**는 뜻이다. negative 건은 교사도
    모델도 `labels`가 비어 자동으로 일치가 되는데, 그것까지 세면 negative가 많을수록
    점수가 올라간다 -- `labels`가 비면 AM2·AM3이 자동 만점이 되는 것과 **똑같은 함정**이다.
    실측으로 `delora`가 전 건 기준 40.5%, 교사가 라벨을 단 건 기준 19.2%였다.

    **순서는 뜻이 없으므로 집합으로 본다.** 같은 라벨을 순서만 바꿔 낸 것을 틀렸다고 하면
    안 된다. `근거`는 자유 문장이라 비교에 안 쓴다 -- 어휘가 고정된 두 칸만 본다.
    """
    theirs = label_pairs(teacher_labels) or []
    if not theirs:
        return None
    pick = PARTS[part]
    return {pick(x) for x in label_pairs(model_labels) or []} == {pick(x) for x in theirs}


def label_agreement(pairs: list[tuple], label_free: bool = False,
                    part: str = "쌍") -> dict:
    """`(모델 labels, 교사 labels)` 짝 목록에서 라벨일치를 센다.

    **`label_free`는 설정에서 와야 한다**(`target == "sentence"`). 출력만 보고 정하면
    안 된다 -- "구조적으로 라벨을 안 내는 조건"과 "붕괴해서 라벨이 안 나온 것"이
    출력 위에서 똑같이 보이기 때문이다. `beft`는 `target=full`인데 홀드아웃 37건에
    전부 negative를 내 라벨이 한 건도 없었다. **그 자리는 해당 없음이 아니라 0%다** --
    해당 없음으로 적으면 붕괴가 표에서 숨는다.

    `rate`가 None이 되는 경우가 둘이고 뜻이 정반대라 `note`로 가른다.

    - `라벨 없는 조건` -- `sentence` 조건이다. **0%가 아니라 해당 없음이다.** 실력이
      아니라 구조이므로, 0%로 적으면 이 조건이 통째로 0점으로 깔린다. AM8s가
      `sentence`에서 자동 탈락을 만드는 것과 같은 자리다(`docs/TODO.md`).
    - `교사 라벨 없음` -- 교사가 어느 건에도 라벨을 안 달았다. 분모가 0이다.

    **이 값은 `verdict`에 안 쓴다.** 문턱(평균 60% · 최저 30%)을 건드리면 라운드끼리
    비교가 끊긴다. `쏠림` 열을 세울 때와 같이 **열만 늘리고 판정은 사람이 읽는다.**

    **그리고 낮은 라벨일치가 곧 못 배웠다는 뜻은 아니다.** 이 과제의 산출물은
    `direct_impact` 문장이고, **라벨이 달라도 그 문장은 얼추 같은 경우가 흔하다.**
    라벨은 그 문장에 이르는 중간 표시라 어느 칸으로 갈랐는지가 갈릴 뿐이다.
    품질 점수가 아니라 눈금으로 읽는다.
    """
    scored = [(m, t) for m, t in pairs if label_pairs(t)]
    if label_free:
        return {"rate": None, "matched": 0, "denominator": len(scored),
                "note": "라벨 없는 조건"}
    if not scored:
        return {"rate": None, "matched": 0, "denominator": 0, "note": "교사 라벨 없음"}
    matched = sum(1 for m, t in scored if label_match(m, t, part))
    return {"rate": round(matched / len(scored), 3), "matched": matched,
            "denominator": len(scored), "note": None}


# `"judgement": "positive"` 처럼 **판정 한 칸만** 뽑는다. 값에 따옴표가 없다고 보고
# 첫 닫는 따옴표까지 읽는다 -- 판정은 어휘가 고정된 칸이라 안전하다.
_JUDGEMENT = re.compile(r'"judgement"\s*:\s*"([^"]*)"')


def salvage_judgement(raw: str) -> str:
    """파싱이 깨진 출력에서 **판정 한 칸만** 건진다. 못 건지면 빈 문자열.

    **AM 점수에는 절대 쓰지 않는다.** AM1은 "출력이 JSON 하나로 파싱된다"이고 그것은
    실제로 실패했으므로 0점이 맞다. 이 함수가 고치는 것은 `교사일치`와 라벨 쪽이다 --
    그쪽은 형식이 아니라 **내용을 재는 눈금**인데, 형식 실패가 그 눈금까지 부순다.

    2026-08-24 2차run에서 드러났다. 파싱 실패 440건이 **전부** 판정을 멀쩡히 들고
    있었고 그중 94%가 교사와 맞았다. 그런데 다섯 항목이 0점이 되면서 교사일치도 같이
    0이 되어, **조건 A가 74.3%로 찍혔다 -- 전부-positive 기준선 83.7%보다 낮다.**
    붕괴가 아닌데 붕괴 신호가 켜진 것이다. 건지면 91.9%가 된다.
    """
    found = _JUDGEMENT.search(raw or "")
    return found.group(1).strip() if found else ""


def salvage_labels(raw: str) -> list | None:
    """파싱이 깨진 출력에서 **`labels` 배열만** 건진다. 못 건지면 None.

    `None`은 "라벨이 비었다"가 아니라 **"이 건은 분모에서 뺀다"**는 뜻이다.
    `label_match`가 교사 라벨이 없을 때 쓰는 뜻과 같다 -- 모델이 무엇이라 했는지
    정말로 모르는 자리이므로, 빈 목록으로 두면 "전부 틀렸다"로 세어진다.

    **대괄호를 셀 때 문자열 안은 안 센다.** `근거`가 자유 문장이라 `[`가 들어갈 수
    있고, 그러면 짝이 어긋나 멀쩡한 배열을 못 건진다.

    실측(2차run 440건): 배열이 JSON으로 읽히는 것 323건(73%), 닫혔는데 JSON이 아닌 것
    70건, 안 닫힌 것 47건. **못 건진 117건은 분모에서 빠지므로 `label_denominator`가
    그만큼 줄어 표에서 보인다.**
    """
    text = raw or ""
    head = text.find('"labels"')
    if head < 0:
        return None
    start = text.find("[", head)
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    found = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return found if isinstance(found, list) else None
    return None


def layered_agreement(records: list[dict], teacher: dict | None = None,
                      label_free: bool = False) -> dict:
    """판정 -> 대상 -> 방향 -> 쌍, **네 층으로 갈라** 센다. 2026-08-24에 더했다.

    지금까지 표에는 `교사일치`(판정 한 칸)와 `라벨일치`(`(대상, 방향)` 쌍) 둘뿐이라,
    라벨일치가 낮을 때 **대상을 못 고른 것인지 방향을 못 고른 것인지 알 수 없었다.**
    2차run에서 갈라 보니 대상 42.4% · 방향 46.0% · 쌍 31.1%로, 둘이 비슷하게 어렵고
    한쪽이 범인이 아니었다.

    **파싱이 깨진 건은 원문에서 건져 쓴다**(`salvage_judgement`·`salvage_labels`).
    형식 실패가 내용 눈금까지 부수기 때문이다 -- 자세한 것은 그 두 함수에 있다.
    **AM 다섯은 이 함수를 안 쓰므로 형식 눈금은 그대로다.**

    파싱이 깨졌는지는 `scores["AM1"]`으로 가른다. AM1이 곧 "출력이 JSON 하나로
    파싱된다"라 다른 신호를 새로 만들 이유가 없다.

    `labels`를 못 건진 건은 **분모에서 뺀다**(`label_unrecoverable`로 몇 건인지 적는다).
    모델이 무엇이라 했는지 정말로 모르는 자리라 0점으로 세면 없는 오답을 만든다.
    **파싱은 됐는데 라벨이 빈 건은 다르다** -- 그건 모델이 실제로 안 낸 것이라 오답으로
    센다.

    기존 `teacher_agreement`·`label_agreement`는 **안 건드린다.** 값이 바뀌면 지난
    147줄 표와 비교가 끊긴다. 이 함수가 내는 것은 전부 새 열이다.
    """
    teacher = teacher or {}
    judged, pairs, salvaged, unrecoverable = [], [], 0, 0
    for record in records:
        raw = record.get("raw") or ""
        theirs = record.get("teacher_labels", teacher.get(record.get("id")))
        if record.get("scores", {}).get("AM1", 1):
            judged.append((record.get("judgement") or "", record.get("teacher_judgement")))
            pairs.append((record.get("labels"), theirs))
            continue
        # 여기부터가 파싱이 깨진 자리다.
        found = salvage_judgement(raw)
        if found:
            salvaged += 1
        judged.append((found, record.get("teacher_judgement")))
        labels = salvage_labels(raw)
        if labels is None:
            unrecoverable += 1
        else:
            pairs.append((labels, theirs))

    matched = sum(1 for mine, theirs in judged if mine == theirs)
    result = {"teacher_agreement_salvaged": round(matched / len(records), 3),
              "salvaged_judgements": salvaged,
              "label_unrecoverable": unrecoverable}
    for key, part in (("target", "대상"), ("direction", "방향"), ("pair", "쌍")):
        counted = label_agreement(pairs, label_free, part)
        result[f"{key}_agreement"] = counted["rate"]
        result[f"{key}_denominator"] = counted["denominator"]
    return result


def impact_subjects(impacts) -> list[str]:
    """impacts 배열에서 주체 문자열만 꺼낸다. 모양이 어긋나면 빈 목록."""
    if not isinstance(impacts, list):
        return []
    return [str(x.get("주체", "")) for x in impacts if isinstance(x, dict)]


def score_blind(raw: str, label_free: bool = False) -> dict:
    """정답키 없이 되는 것만 매긴다. AM4·AM5·AM7은 사람 라벨이 있어야 하므로 없다.

    **파싱이 깨지면 다섯 개가 전부 0점이다.** 첫 관문에서 되돌아 나가기 때문이다.
    그래서 다섯 항목 평균은 사실상 파싱률을 따라가고, 실패 기준의 숫자가 작동하는
    이유도 이것이다.

    **2026-08-27 -- `labels`가 비면 AM2·AM3에 만점을 주지 않는다.** 그전에는 AM2가
    검사할 어휘가 없어 자동 1점, AM3은 0개라 중복이 없어 자동 1점이었다. 그래서
    **"전부 negative에 빈 배열"이 다섯 항목 만점을 받았다** -- `collapsed()`의 설명이
    2026-08-11부터 "제일 게으른 답이 만점 전략이다"라고 적어 둔 그 자리다.

    **문턱을 새로 만든 것이 아니라 이미 있던 원리를 옮긴 것이다.** AM8s가 벌써
    똑같이 한다 -- "positive인데 배열이 비면 검사할 것이 없어 공짜 점수가 되므로
    0으로 막는다". 검사할 것이 없을 때 만점을 안 주는 것이 원래 규칙이었고, AM2·AM3
    두 칸에만 안 걸려 있었다.

    왜 이날 필요해졌나. `parse_output`을 갈아 끼우자 **135건 중 반대 판정 1건**이
    건져지면서 `collapsed()`가 꺼졌고(판정이 두 종류가 됐다), 게으른 답을 걸러 주던
    장치가 그것 하나뿐이라 `lora-lr1e-4`·`lora-lr5e-5`·`lora-r8`·`lora-full-bare-s44-r2`
    넷이 **교사일치 17%로 `됨` 판정을 받게 됐다**(무조건 negative의 값이 16.3%다).
    이 수정으로 그 넷의 AM 최저가 0.007로 내려가 `verdict()`가 스스로 거른다.

    `label_free`는 **`labels`를 아예 안 내는 것이 정상인 조건**(`target: sentence`)이다.
    그쪽은 빈 배열이 게으름이 아니라 설계이므로 이 규칙에서 뺀다.

    ── 원본과 갈라진 자리 ──────────────────────────────────────────────
    **AM2·AM3의 정의가 `data_collect`의 `rubric.md`와 달라졌다.** 파서가 갈라진 것과는
    성격이 다르다 -- 파서는 같은 눈금으로 더 많이 읽는 것이었고, 이것은 **눈금 자체가
    다른 것이다.** 두 벌의 AM2·AM3을 한 표에 놓으면 안 된다.
    """
    result = {"AM1": 0, "AM2": 0, "AM3": 0, "AM6s": 0, "AM8s": 0}
    parsed = parse_output(raw)
    if parsed is None:
        return {**result, "parsed": None}
    result["AM1"] = 1

    pairs = label_pairs(parsed.get("labels", []))
    if pairs is None:
        return {**result, "parsed": parsed}
    if not pairs and not label_free:
        # 검사할 것이 없다. AM8s와 같은 이유로 공짜 점수를 안 준다.
        result["AM2"] = result["AM3"] = 0
    else:
        result["AM2"] = int(all(t in TARGETS and d in DIRECTIONS for t, d in pairs))
        result["AM3"] = int(len(pairs) == len(set(pairs)))

    judgement = str(parsed.get("judgement", "")).strip()
    subjects = impact_subjects(parsed.get("impacts"))
    sentence = str(parsed.get("direct_impact") or "")

    # AM6s -- 스스로 negative라 해놓고 impacts나 문장을 채웠으면 자기모순이다.
    if judgement == "negative":
        result["AM6s"] = int(not subjects and not sentence.strip())
    else:
        result["AM6s"] = 1

    # AM8s -- 자기가 낸 주체를 자기 문장에서 흘리지 않았나. positive인데 배열이 비면
    # 검사할 것이 없어 공짜 점수가 되므로 0으로 막는다(run.py의 AM8과 같은 이유).
    if judgement == "positive" and not subjects:
        result["AM8s"] = 0
    else:
        result["AM8s"] = int(all(s in sentence for s in subjects if s))
    return {**result, "parsed": parsed}


def restatement_ratio(after: str, sentence: str) -> float | None:
    """해설이 개정문을 그대로 옮긴 것인지 보는 선별기. 합격 판정이 아니라 걸러내기다."""
    if not sentence.strip():
        return None
    return round(difflib.SequenceMatcher(
        None, sentence, after, autojunk=False).ratio(), 3)


def collapsed(judgements: list[str]) -> bool:
    """모든 항목에 같은 판정을 내놓았는가. **채점 점수와 별개로 본다.**

    2026-08-11에 기준 조건 첫 판에서 드러난 구멍이다. 학습된 모델이 37건 전부에
    이 답을 냈다.

        {"judgement":"negative","labels":[],"impacts":[],"direct_impact":""}

    **이 답은 다섯 항목 만점을 받는다.** `labels`가 비면 AM2는 검사할 어휘가 없어
    자동 1점, AM3은 0개라 중복도 0이라 자동 1점이다. AM6s는 negative에 배열이
    비었으니 자기일관이고, AM8s는 주체가 없으니 흘릴 것도 없다. **제일 게으른 답이
    만점 전략이다.**

    그런데 홀드아웃 37건 중 26건이 positive였으므로 실제로는 26건을 틀렸다.

    학습이 이쪽으로 가는 이유도 분명하다 -- 학습 562건 중 258건이 negative이고
    그 정답이 23토큰으로 제일 짧아, 손실을 제일 빨리 줄이는 길이 "전부 negative"다.

    **AM 값 자체는 건드리지 않는다.** 교사 값과 라운드끼리의 비교가 끊기기 때문이다.
    이 검사는 그 옆에 따로 세워 둔다.

    **2026-08-12 수정 -- 파싱 실패를 판정으로 세지 않는다.** 첫 판에서는 종류를
    그냥 세었는데, 그러면 `{'': 1, 'negative': 36}`이 "두 종류"가 되어 검사를
    빠져나갔다. 빈 문자열은 모델이 내린 판정이 아니라 JSON을 못 읽었다는 표시라
    판정 축에 세우면 안 된다. 실제로 psoft·road·hra·loha·lokr 다섯이 이 구멍으로
    `됨` 판정을 받았고, 교사 일치는 전부 27~30%(=무조건 negative의 값)였다.

    **기준을 낮춘 것이 아니라 원래 뜻대로 되돌린 것이다.** 이 함수는 처음부터
    "판정이 한 종류뿐인가"였고, 파싱 실패는 판정이 아니다. 결과를 보고 문턱을
    맞춘 것이 아니므로 `verdict`에 걸린 금기(아래)에 해당하지 않는다.
    """
    said = [j for j in judgements if j]
    return len(said) > 1 and len(set(said)) == 1


def skew(judgements: list[str]) -> float | None:
    """제일 많이 낸 판정이 읽힌 것 중 차지하는 몫. **합격 판정에 쓰지 않는다.**

    `collapsed`는 예·아니오라 "37건 중 35건이 negative" 같은 **거의 붕괴**를 못 잡는다.
    그렇다고 여기에 문턱을 세우면 결과를 보고 기준을 만드는 것이 되므로, 숫자만
    표에 세워 두고 판정은 사람이 한다. 읽힌 것이 없으면 None -- 붕괴가 아니라 고장이다.
    """
    said = [j for j in judgements if j]
    if not said:
        return None
    return round(max(said.count(j) for j in set(said)) / len(said), 3)


def verdict(rates: dict[str, float]) -> str:
    """`돌리기 전에 고정된` 실패 기준으로 한 라운드를 판정한다.

    **이 함수를 결과 보고 고치지 않는다.** 나온 것을 보고 기준을 맞추면 라운드끼리
    비교가 안 된다(`rubric.md` 첫 문단). 스무 개 조합을 돌리면 애매한 것이 반드시
    나오는데 거기서 문턱을 손대면 표 스무 개가 전부 못 쓰게 된다.

    `못 돌림`(OOM·예외·학습 미완료)은 점수가 아예 없는 경우라 여기서 판정하지 않는다.
    호출하는 쪽이 학습이 끝까지 갔는지 먼저 보고 이 함수를 부른다.
    """
    if not rates:
        return "이상함"
    values = [rates[k] for k in KEYS if k in rates]
    if len(values) != len(KEYS):
        return "이상함"
    mean = sum(values) / len(values)
    return "됨" if mean >= 0.60 and min(values) > 0.30 else "이상함"
