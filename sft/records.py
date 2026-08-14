"""교사 해석 레코드에서 **무엇을 버리고 무엇을 평가용으로 뺄지**를 정한다.

`data_collect`가 실제 문서에서 뽑은 실질 변경에 교사 모델(Solar)의 해석을 붙여 놓은
것이 입력이다. 여기서 하는 일은 거르기와 가르기 둘뿐이다.

**프롬프트와 정답 문자열은 여기서 만들지 않는다.** 규칙서를 붙일지, 정답을 어디까지
둘지는 실험 조건이라 `sft.formatting`이 학습·평가 시점에 조립한다. 데이터를 한 벌로
얼려두어야 두 실험이 정말 같은 데이터를 썼는지 파일 해시로 확인된다.

**negative를 기본으로 남긴다.** 피어 세션은 빼라고 했지만 기획서 3.3 축 2가
'Negative 사례 제거'를 실험 조건으로 적어두었다 -- 처음부터 빼면 그 실험을 못 한다.
채점 항목 AM6s가 재는 것도 정확히 "negative일 때 입을 다무는가"인데, negative를 한 건도
안 본 모델이 그것을 할 리가 없다. **넣어두면 나중에 뺄 수 있지만, 안 넣어두면 만들 수
없다.**

2026-08-12에 실제로 확인됐다. negative를 뺀 조건(`lora-full-nonegative`)은 세 판 다
반대쪽(무조건 positive)으로 붕괴했다. `docs/스윕_결과.md` 5절에 있다.
"""
from __future__ import annotations

from collections import Counter

# 평가용으로 통째로 빼두는 계열. 학습에 쓴 것으로 채점하면 점수가 부풀려진다. 해양수산
# 운영규정을 고른 이유는 37건뿐이라 학습 손해가 작으면서, 처리방침과 문체·형식이 전혀
# 달라 "처음 보는 문서에서도 되는가"를 실제로 재기 때문이다.
HOLDOUT_SERIES = "mof_rd_regulation_pair"

# 프롬프트를 깎을 때 쓴 11건. 학습에 들어가면 그 프롬프트로 잰 값이 무의미해진다.
# `evalset.json`은 `data_collect`에 있고 이 저장소에는 없으므로 걸러낼 `before_id`만
# 옮겨 적어 둔다. 11개뿐이고 늘어날 것이 아니라 파일을 통째로 들고 올 이유가 없다.
EVALSET_BEFORE_IDS = {
    "before_2022-B0618", "before_2022-B0817", "2020_standard_terms.converted-B0056",
    "mobile_2017.converted-B0088", "privacyOld14-B0885", "privacyOld14-B0117",
    "before_2022-B2063", "privacyOld14-B0870", "privacyOld11-B0930",
    "before_2026-40-B0003", "privacyOld12-B0610",
}

# 학습·평가에 실제로 쓰는 칸만 남긴다. 교사의 `raw`(원문 응답)와 `scores`는 이 저장소
# 에서 쓸 일이 없고, 셋을 다 두면 파일이 세 배가 된다.
FIELDS = ("id", "series", "before_id", "after_id", "before", "after",
          "judgement", "labels", "impacts", "direct_impact")


def usable(record: dict) -> bool:
    """학습 레코드가 될 수 있는가.

    `annotate.py`가 막 뱉은 것과, `data_collect`가 품질 필터까지 걸어 넘긴 것 두 가지가
    들어온다. **뒤엣것에는 `scores`가 없다** -- 이미 걸러진 뒤라 채점 결과를 들고 다닐
    이유가 없기 때문이다. `scores`가 있으면 여기서 한 번 더 거르고, 없으면 걸러진
    것으로 보고 판정만 확인한다.

    교사 호출이 실패했거나(`error`), 교사 출력이 JSON으로 안 읽힌 것(`AM1`이 0)은
    정답이 없으므로 쓸 수 없다.

    **`scores`가 있어도 `AM*`가 하나도 없으면 거르지 않는다.** `data_collect`의 생성
    쪽(역할 B) `scores`에는 `BM1~BM7`만 들어 있다. `"scores" in record`만 보고 걸렀다면
    `scores.get("AM1")`이 `None`이라 **넘어온 것이 전량 조용히 버려진다** -- 에러가 안
    나고 건수만 0이 되는 종류의 사고다. 지금은 최종 파일에 `scores`를 안 담기로
    정리돼 있지만, 그것은 남의 저장소의 운영이지 이쪽이 보장할 수 있는 것이 아니다.
    """
    if "error" in record:
        return False
    scores = record.get("scores")
    if scores and any(k.startswith("AM") for k in scores) and not scores.get("AM1"):
        return False
    return record.get("judgement") in {"positive", "negative"}


def split(records: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    """학습용·평가용으로 가르고, 버린 것은 이유별로 센다.

    **버린 이유를 세는 것이 중요하다.** 605건이 562건이 된 것은 정상이지만, 605건이
    300건이 되었다면 교사 쪽에 문제가 생긴 것이다. 숫자만 보고는 못 가린다.
    """
    train, holdout, dropped = [], [], Counter()
    for record in records:
        if not usable(record):
            dropped["교사 출력을 못 씀"] += 1
            continue
        if record["before_id"] in EVALSET_BEFORE_IDS:
            dropped["평가셋 11건과 겹침"] += 1
            continue
        slim = {k: record.get(k) for k in FIELDS}
        # **합성분은 계열에 `synth:` 접두가 붙어 온다** -- `synth:mof_rd_regulation_pair`.
        # 정확일치로 두면 그것이 이 필터를 그냥 통과해 학습셋으로 들어간다. 합성 pair는
        # 원천 조항을 그대로 가져다 개정문만 새로 쓰므로, **채점에 쓰는 바로 그 조항으로
        # 학습하게 되고 점수가 부풀려진다.** 접두를 떼고 비교한다.
        #
        # `slim`에는 접두가 붙은 원래 값을 그대로 남긴다. 어느 것이 합성분인지는
        # 나중에 계열별로 세어 볼 때 필요한 정보다.
        series = record["series"].removeprefix("synth:")
        (holdout if series == HOLDOUT_SERIES else train).append(slim)
    return train, holdout, dropped
