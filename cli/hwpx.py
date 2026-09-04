"""HWPX를 문단 목록으로 펼치고, 개정 전후를 맞대어 **바뀐 구간만** 고른다.

**`master` 가지의 `source_data/extract.py`에서 옮겨 심었다.** `sft/scoring.py`와 같은
사정이다 -- 저쪽은 문서를 모으는 저장소고 이쪽은 학습·추론 저장소인데, 실습 페이지가
파일을 직접 받으려면 읽는 코드가 이쪽에 있어야 한다.

**옮기면서 한 군데를 고쳤다.** 원본 `blocks(path: Path)`가 여기서는 `blocks(data:
bytes)`다. 서버가 업로드를 메모리에서 처리하므로 임시 파일을 만들 이유가 없고,
`zipfile`은 `io.BytesIO`를 파일처럼 받는다. **문단을 끊는 규칙은 한 글자도 안 바꿨다** --
바꾸면 여기서 나온 블록과 학습 데이터의 블록(`...-B0056`)이 다른 물건이 된다.

**갈라질 수 있는 자리는 `_paragraph_text`와 `_clean` 둘이다.** 저쪽이 문단 규칙을 고치면
같은 문서에서 다른 블록이 나온다. `docs/TODO.md`에 적어 두었다.

**HWPX만 읽는다.** 구형 `.hwp`는 zip이 아니라 OLE 복합문서라 표준 라이브러리로 못
읽는다(`olefile` 같은 새 패키지가 필요하고 이 저장소는 판본을 안 올린다). 저쪽
`extract.py`도 "HWP 5.0 has no extractor yet"이다. 원본이 같이 읽던 html·txt 갈래는
쓸 데가 없어 안 가져왔다.

**새 패키지를 안 쓴다.** HWPX가 사실 zip이라 `zipfile`·`xml.etree`로 열리고, 맞대는
것은 `difflib`이다. 셋 다 파이썬에 딸려 온다.
"""
from __future__ import annotations

import io
import re
import zipfile
from difflib import SequenceMatcher
from xml.etree import ElementTree

HWPX_NS = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"

# 구간 하나가 이보다 길면 표시만 달아 둔다. **어댑터가 4,096토큰까지만 학습됐고**
# 규칙서가 약 1,600토큰, 답에 768토큰을 남기므로 본문 몫이 한글 3,600자쯤이다.
# 실측한 문서 9쌍의 구간 96개에서 제일 긴 것이 972자라 평소에는 안 걸린다.
MAX_REGION_CHARS = 3000


def _clean(lines) -> list[str]:
    out = [re.sub(r"\s+", " ", x).strip() for x in lines]
    return [x for x in out if x]


def _paragraph_text(para) -> str:
    """`<hp:p>` 하나의 글자. **안에 겹쳐 든 문단은 뺀다.**

    표가 문단 안에 들어앉는 구조라(`hp:p > hp:tbl > hp:tr > hp:tc > hp:subList >
    hp:p`) 자손을 통째로 훑으면 표의 모든 칸이 그 표를 담은 문단 하나로 빨려 든다.
    칸에 든 문단은 부르는 쪽이 문서 순서대로 따로 받는다.

    한 문단 안에서도 글꼴이 바뀌는 자리마다 `hp:run`이 갈린다 -- 굵은 `제2조
    (용어의 정의)`와 뒤따르는 보통 굵기의 `① ...`이 한 문장의 두 run이다 -- 그래서
    run들은 따로 떼지 않고 이어 붙인다.
    """
    parts: list[str] = []

    def walk(node) -> None:
        for child in node:
            if child.tag == HWPX_NS + "p":
                continue  # 겹쳐 든 문단 몫이지 이 문단 몫이 아니다
            if child.tag == HWPX_NS + "t":
                parts.append("".join(child.itertext()))
            walk(child)

    walk(para)
    return "".join(parts)


def blocks(data: bytes) -> list[str]:
    """HWPX 한 벌을 문단 단위 글자 덩어리로 펼친다.

    `zipfile.BadZipFile`을 그대로 올려보낸다. **부르는 쪽이 "hwpx가 아니다"라고
    사람 말로 바꿔 주어야 하기 때문이다** -- 여기서 삼키면 빈 목록과 구분이 안 된다.
    """
    out: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        sections = sorted(
            name for name in archive.namelist()
            if re.search(r"Contents/section\d+\.xml$", name)
        )
        for name in sections:
            root = ElementTree.fromstring(archive.read(name))
            for para in root.iter(HWPX_NS + "p"):
                out.append(_paragraph_text(para))
    return _clean(out)


def similarity(before, after) -> float:
    """두 벌이 얼마나 닮았나. 0에 가까우면 남남, 1에 가까우면 같은 글이다.

    **문단 목록과 글자열 둘 다 받는다.** 문서끼리 맞댈 때는 문단 목록을,
    조문 하나끼리 맞댈 때는 글자열을 넘긴다 -- `SequenceMatcher`가 어느 쪽이든
    "같은 자리가 얼마나 되나"로 답한다.
    """
    return SequenceMatcher(None, before, after, autojunk=False).ratio()


def changed_regions(before: list[str], after: list[str],
                    max_chars: int = MAX_REGION_CHARS) -> list[dict]:
    """두 문단 목록을 맞대어 **바뀐 구간만** 돌려준다.

    **문단 번호로 짝지으면 안 된다.** 실제 문서 한 쌍이 개정 전 77문단 · 개정 후
    80문단인데, 가운데에 세 문단이 끼면 그 뒤가 전부 한 칸씩 밀린다. 번호로 맞대면
    밀린 자리부터 끝까지 엉뚱한 문단끼리 비교하게 된다.

    `SequenceMatcher`는 **끼어든 자리를 알아보고** 그 앞뒤를 다시 맞춘다. 위의 쌍에서
    바뀐 구간 3개만 남고, 1문단이 4문단으로 늘어난 삽입도 한 구간으로 묶인다.

    구간 하나가 `max_chars`를 넘으면 `oversize`를 달아 둔다. **막지는 않는다** --
    페이지가 체크를 풀어 두고 이유를 적게 하는 표시일 뿐이고, 사람이 굳이 물어보겠다면
    말리지 않는다.
    """
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    regions = []
    for kind, i1, i2, j1, j2 in matcher.get_opcodes():
        if kind == "equal":
            continue
        old, new = "\n".join(before[i1:i2]), "\n".join(after[j1:j2])
        regions.append({
            "kind": kind,
            "before": old,
            "after": new,
            # 문서에서 몇 번째 문단이었는지. 1부터 센다 -- 저쪽 `indexed()`가 붙이는
            # 블록 번호(`before-B0001`)와 같은 기준이라 눈으로 대조할 수 있다.
            "before_at": [i1 + 1, i2],
            "after_at": [j1 + 1, j2],
            "chars": len(old) + len(new),
            "oversize": len(old) + len(new) > max_chars,
        })
    return regions
