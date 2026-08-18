"""학생 모델(KORMo) 출력을 정답키 없이 채점한다. `data_collect`에서 옮겨 심은 것이다.

**왜 옮겨 심었나.** 원본은 `data_collect/training_data/interpret/`의 `annotate.py`와
`run.py`에 있는데, 두 파일 다 맨 위에서 `solar`를 import 한다. `solar.py`는 교사 모델
API를 부르는 모듈이라 API 키를 요구하고, 학습 서버에는 그 레포 자체가 없다. 채점에
실제로 쓰이는 것은 순수 함수 몇 개뿐이므로 그것만 가져왔다.

**한 글자도 바꾸지 않았다.** 채점 로직이 갈라지면 교사 값과 학생 값을 한 표에
못 놓는다. `rubric.md`가 적어둔 사고 -- 이름이 같은데 잣대가 다르면 언젠가 누군가
반드시 한 줄에 놓는다 -- 가 여기서도 그대로 성립한다. 원본이 바뀌면 이쪽도 바꾸고,
`test_scoring.py`가 두 벌이 같은 답을 내는지 대조한다.

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

# rubric.md AM2가 쓰는 어휘 목록. 이 밖의 말이 하나라도 나오면 0점이다.
TARGETS = ["기한·시점", "수치·기준", "적용 범위", "수행 주체",
           "절차·요건", "제출물·기재사항", "명칭"]
DIRECTIONS = ["늘었다", "줄었다", "다른 값", "새로 생겼다", "없어졌다"]

# 채점 항목 이름. `s`는 self-consistency로, 정답키 대신 모델 자기 판정으로 분기한다는
# 뜻이다. 평가 세트의 AM6·AM8과 잣대가 다르므로 나란히 놓지 않는다(rubric.md).
KEYS = ("AM1", "AM2", "AM3", "AM6s", "AM8s")

# AH1(재진술 아님)을 사람이 읽기 전에 명백한 복사를 걸러내는 값. 합격 판정에는 쓰지
# 않는다 -- 유사도가 낮은 실패가 실제로 있다. 짧은 블록에서 오탐이 나는 것도 확인돼
# 있으므로(rubric.md), 이 값으로 점수를 매기지 말고 분포만 본다.
RESTATEMENT_THRESHOLD = 0.60


def parse_output(text: str) -> dict | None:
    """모델 출력에서 JSON 객체 하나를 꺼낸다. 코드펜스는 벗긴다."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.split("\n", 1)[1] if cleaned.startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        result = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


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


def label_match(model_labels, teacher_labels) -> bool | None:
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
    return set(label_pairs(model_labels) or []) == set(theirs)


def label_agreement(pairs: list[tuple], label_free: bool = False) -> dict:
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
    matched = sum(1 for m, t in scored if label_match(m, t))
    return {"rate": round(matched / len(scored), 3), "matched": matched,
            "denominator": len(scored), "note": None}


def impact_subjects(impacts) -> list[str]:
    """impacts 배열에서 주체 문자열만 꺼낸다. 모양이 어긋나면 빈 목록."""
    if not isinstance(impacts, list):
        return []
    return [str(x.get("주체", "")) for x in impacts if isinstance(x, dict)]


def score_blind(raw: str) -> dict:
    """정답키 없이 되는 것만 매긴다. AM4·AM5·AM7은 사람 라벨이 있어야 하므로 없다.

    **파싱이 깨지면 다섯 개가 전부 0점이다.** 첫 관문에서 되돌아 나가기 때문이다.
    그래서 다섯 항목 평균은 사실상 파싱률을 따라가고, 실패 기준의 숫자가 작동하는
    이유도 이것이다.
    """
    result = {"AM1": 0, "AM2": 0, "AM3": 0, "AM6s": 0, "AM8s": 0}
    parsed = parse_output(raw)
    if parsed is None:
        return {**result, "parsed": None}
    result["AM1"] = 1

    pairs = label_pairs(parsed.get("labels", []))
    if pairs is None:
        return {**result, "parsed": parsed}
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
