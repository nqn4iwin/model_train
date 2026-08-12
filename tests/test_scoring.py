"""옮겨 심은 채점기가 원본과 같은 답을 내는지 대조한다.

**원본을 import 하지 않는다.** `annotate.py`는 맨 위에서 `solar`를 부르고, 그것은 API
키와 `data_collect` 레포를 요구한다. 학습 서버에는 둘 다 없으므로 import 대조는
서버에서 돌아가지 않는다.

대신 **원본이 이미 매겨 놓은 점수와 맞춰 본다.** `annotate.py`는 레코드마다 모델
원문(`raw`)과 자기가 매긴 점수(`scores`)를 함께 남긴다. 옮겨 심은 `score_blind()`에
같은 `raw`를 넣어 같은 `scores`가 나오면, 두 벌이 같은 잣대라는 것이 실제 출력
수백 건 위에서 확인된다. import 가 되는지보다 이쪽이 강한 증거다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m tests.test_scoring <records.jsonl 경로>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sft.scoring import KEYS, score_blind


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    path = Path(sys.argv[1])

    checked = mismatched = skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        # 호출이 실패한 레코드에는 raw도 scores도 없다. 셀 것이 없으므로 넘긴다.
        if "scores" not in record or "raw" not in record:
            skipped += 1
            continue
        checked += 1
        theirs = {k: record["scores"][k] for k in KEYS}
        ours = score_blind(record["raw"])
        ours.pop("parsed")
        if ours != theirs:
            mismatched += 1
            print(f"  어긋남  {record.get('id')}")
            for key in KEYS:
                if ours[key] != theirs[key]:
                    print(f"    {key}  원본 {theirs[key]} -> 옮긴 것 {ours[key]}")

    print(f"\n대조 {checked}건 · 어긋남 {mismatched}건 · 건너뜀 {skipped}건")
    if mismatched:
        raise SystemExit("채점기가 갈라졌습니다. 옮겨 심은 쪽을 원본에 맞추세요.")
    # **한 건도 못 댔으면 통과라고 하면 안 된다.** `data_collect`가 품질 필터까지 걸어
    # 넘긴 파일에는 `raw`도 `scores`도 없어서 여기가 조용히 0건이 된다. 그것을 "같은
    # 잣대입니다"로 찍으면 확인한 적 없는 것을 확인했다고 믿게 된다.
    if not checked:
        raise SystemExit(
            f"대조할 것이 없습니다 ({skipped}건 전부 raw나 scores가 없음).\n"
            "  `annotate.py`가 막 뱉은 원본 records.jsonl을 주세요. 걸러진 뒤의\n"
            "  파일에는 대조에 필요한 두 칸이 빠져 있습니다.")
    print("두 벌이 같은 잣대입니다.")


if __name__ == "__main__":
    main()
