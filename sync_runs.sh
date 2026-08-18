#!/usr/bin/env bash
# 서버의 실험 결과 중 **읽을 수 있는 것만** 로컬로 당겨온다.
#
# 체크포인트(학습된 어댑터 가중치)는 안 가져온다. 하나가 수백 MB이고 68개 실험에
# 에폭별로 셋씩 있어서, 다 받으면 수십 GB가 된다. 그것은 서버에 두고 필요할 때만
# 골라서 받는다.
#
# rsync는 바뀐 부분만 골라 복사하는 도구다. 같은 명령을 몇 번 다시 돌려도 이미 받은
# 파일은 건너뛰므로, 스윕이 도는 동안 몇 분마다 돌려도 부담이 없다.
#
# 사용:
#     bash sync_runs.sh
set -euo pipefail

SERVER="${SERVER:-ad-068}"
REMOTE="${REMOTE:-/data1/yblee/repository/model_train/runs/}"
LOCAL="$(cd "$(dirname "$0")" && pwd)/runs/"

mkdir -p "$LOCAL"

# --include 를 --exclude 보다 먼저 쓴다. rsync는 위에서부터 처음 맞는 규칙을 따르므로,
# 가져올 것을 먼저 적고 맨 아래에서 나머지를 통째로 막는다.
# **`sweep.md`·`sweep.json`은 일부러 안 가져온다.** 표를 짓는 주체가 로컬이기 때문이다.
# `cli.sweep`은 끝날 때 **이번에 넘긴 설정만** 표에 쓰므로, 서버의 표는 그 라운드 몇 줄
# 짜리 토막이다. 그걸 당겨오면 로컬의 140줄이 그 토막으로 덮이고, `못 돌림` 줄은
# `records.jsonl`이 없어 **되살릴 방법이 없다.**
#
#     서버   records.jsonl 을 만든다
#     로컬   그것을 모아 `cli.rescore --write` 로 표를 짓는다
#
# 서버의 표를 보고 싶으면 거기서 `cat runs/sweep.md` 한다.
#
# --update: **받는 쪽이 더 새 파일이면 건너뛴다.** 이게 없으면 로컬에서 만든 것이
# 서버의 옛 판으로 덮인다 -- 2026-08-18에 `runs/sweep.json`이 그럴 뻔했다. 로컬 것은
# `cli.rescore --write`로 두 라운드를 합쳐 라벨일치까지 넣은 140줄인데, 서버 것은
# 2차만 담긴 66줄이라 그대로 덮였으면 그날 작업이 통째로 날아간다.
rsync -avz --update --prune-empty-dirs \
  --include='*/' \
  --include='*.log' \
  --include='summary.json' \
  --include='records.jsonl' \
  --include='config.json' \
  --exclude='*' \
  "$SERVER:$REMOTE" "$LOCAL"

echo
echo "받은 것:"
du -sh "$LOCAL" 2>/dev/null || true
# 표는 여기서 안 찍는다 -- 140줄이라 매번 흘러가고, 그 표는 sync 가 아니라
# `cli.rescore --write` 가 짓는다. 새 결과를 표에 넣으려면:
echo
echo "표에 넣으려면:  python -m cli.rescore        (보여주기만)"
echo "                python -m cli.rescore --write (표를 다시 씀)"
