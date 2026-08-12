"""명령줄에서 부르는 것들. **저장소 뿌리에서 `-m`으로 부른다.**

    python -m run.train --config configs/delora.json

`python run/train.py` 로는 안 된다. 그렇게 부르면 파이썬이 `run/`을 기준으로 모듈을
찾아 `sft`를 못 보기 때문이다. `-m`은 지금 있는 폴더를 기준으로 삼는다.
"""
