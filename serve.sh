#!/usr/bin/env bash
# 실습 서버를 띄운다. **서버에서 돌린다.**
#
# venv · GPU 자리 · 바깥 개방 셋을 한 줄로 묶는다. 하나씩 빠뜨렸을 때 **막히는
# 방식이 제각각이라** 매번 원인을 다시 찾게 되기 때문이다.
#
#     venv 를 안 켜면            `No module named 'torch'`
#     CUDA_VISIBLE_DEVICES 없으면  실행을 거부한다 (남의 0번 GPU 를 잡지 않으려고)
#     --host 를 안 주면           나만 볼 수 있다 (127.0.0.1 로만 열린다)
#
# **로그인 장치가 없다.** `--host 0.0.0.0` 은 그 포트에 닿는 사람 누구에게나 이 GPU 를
# 내주는 것이므로, 닿을 사람이 정해진 망에서만 쓴다.
#
# 사용:
#     bash serve.sh
#     GPUS=2,3 bash serve.sh            # 우리 자리가 또 옮겨졌을 때
#     PORT=9000 bash serve.sh           # 8137 도 이미 쓰이고 있을 때
#     bash serve.sh --only kormo-good   # 남은 인자는 cli.serve 로 그대로 간다
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# `python -u` 는 출력을 모아 두지 말라는 뜻이다. 안 붙이면 `> log` 로 돌렸을 때
# 로그가 한참 안 쌓여 죽었는지 도는지 알 수 없다.
export CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
exec python -u -m cli.serve --host 0.0.0.0 --port "${PORT:-8137}" "$@"
