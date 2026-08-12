"""레코드 하나를 프롬프트와 정답 문자열로 조립한다.

**왜 데이터 파일에 미리 박지 않는가.** 규칙서가 7KB라, 레코드마다 붙여 두면 430건짜리
파일이 3.6MB가 되고 조건이 스무 개면 거의 같은 파일이 스무 벌 생긴다. 무엇보다
**조건이 데이터에 박히면 "다른 건 똑같이 두고 하나만 바꾼다"가 눈으로 확인이 안 된다.**
데이터는 한 벌로 얼려두고 조건은 설정에 두면, 두 실험이 정말 같은 데이터를 썼는지가
파일 해시 하나로 결정된다.

**필드 순서를 바꾸지 않는다.** `labels -> impacts -> direct_impact`는 자기회귀 모델에서
앞 필드가 뒤 필드의 입력이 되게 하려는 설계다(Distilling Step-by-Step). 경위는
`data_collect/training_data/설계_메모.md` 2절에 있다.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

# 이 파일은 sft/ 안에 있고 규칙서는 저장소 뿌리의 prompts/ 에 있으므로 한 단계 올라간다.
ROOT = Path(__file__).resolve().parents[1]

TARGETS = ("full", "no-impacts", "sentence")

# 규칙서를 안 붙이는 조건에서 쓰는 최소 지시. 아무 말도 없이 블록쌍만 주면 모델이 무슨
# 일을 해야 하는지 모른다. 규칙을 가르치지 않으면서 과제만 알리는 선이다.
BARE_INSTRUCTION = """다음 공공문서 조항의 개정 전후를 대조해, 무엇이 어떻게 바뀌었고
누구의 처지가 달라졌는지 JSON 하나로 출력하세요.

[이전판] $before_id
$before

[최신판] $after_id
$after
"""

_cache: dict[str, Template] = {}


def template(rules: bool | str) -> Template:
    """규칙서를 붙일지 고른다. 문자열을 주면 `prompts/<그 이름>.txt`를 쓴다.

    제로샷 베이스라인에는 규칙서가 필수다 -- 학습이 없으니 대상 7종·방향 5종이라는
    어휘 자체를 모른다. SFT에서는 예시로 익히는 것이 base 모델에 맞을 수 있어 조건으로
    갈라 둔다. 붙이면 4K 시퀀스의 절반가량을 규칙서가 차지한다.
    """
    name = "roleA_v2.2" if rules is True else rules
    if not name:
        return Template(BARE_INSTRUCTION)
    if name not in _cache:
        _cache[name] = Template(
            (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8"))
    return _cache[name]


def build_prompt(record: dict, rules: bool | str = True) -> str:
    return template(rules).substitute(
        before_id=record["before_id"], before=record["before"],
        after_id=record["after_id"], after=record["after"])


def build_completion(record: dict, target: str = "full") -> str:
    """정답으로 삼을 필드를 고른다. **순서는 어느 조건에서도 그대로 둔다.**

    full        judgement -> labels -> impacts -> direct_impact   교사가 낸 그대로
    no-impacts  judgement -> labels -> direct_impact              배열만 뺀다
    sentence    judgement -> direct_impact          기획서 3.3 축 1의 '최종 답변만'

    **`sentence` 조건은 채점에서 공짜 점수를 받는다.** `labels`가 아예 없으면 AM2(어휘
    준수)와 AM3(중복 없음)이 검사할 것이 없어 둘 다 1점이 된다. 다섯 항목 중 둘이
    구조적으로 유리하므로, 축 1의 두 조건을 나란히 놓을 때 이것을 감안해야 한다.
    """
    if target not in TARGETS:
        raise ValueError(f"target은 {TARGETS} 중 하나여야 합니다: {target!r}")
    answer: dict = {"judgement": record["judgement"]}
    if target in {"full", "no-impacts"}:
        answer["labels"] = record.get("labels") or []
    if target == "full":
        answer["impacts"] = record.get("impacts") or []
    answer["direct_impact"] = record.get("direct_impact") or ""
    return json.dumps(answer, ensure_ascii=False)
