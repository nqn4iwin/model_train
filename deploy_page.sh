#!/usr/bin/env bash
# 실습 페이지를 다시 짓고 서버로 올린다. **로컬(WSL)에서 돌린다.**
#
# 페이지를 짓는 주체가 로컬인 이유는 `sync_runs.sh` 머리말과 같다 -- 서버는 재료를
# 만들고 로컬이 그것으로 결과물을 짓는다. 게다가 `visualizations/` 는 `.gitignore` 에
# 들어 있어 `git push` 로는 안 따라간다. 그래서 이 스크립트가 필요하다.
#
# **페이지만 바뀌었으면 서버를 안 내려도 된다.** `cli/serve.py` 가 요청이 올 때마다
# HTML 을 새로 읽으므로, 올린 뒤 브라우저를 새로 고치면 바로 새 화면이다.
# **`cli/*.py` 가 바뀌었으면 내렸다 올려야 한다** -- 이미 뜬 프로세스는 옛 코드를
# 물고 있다. 그때는 모델을 다시 싣느라 몇 분 걸린다.
#
# 사용:
#     bash deploy_page.sh
#     SERVER=다른주소 bash deploy_page.sh
set -euo pipefail

SERVER="${SERVER:-ad-068}"
REMOTE="${REMOTE:-/data1/yblee/repository/model_train}"
PYTHON="${PYTHON:-python3}"
PAGE="visualizations/정리_1_결과.html"

cd "$(dirname "$0")"

# 1. 페이지를 다시 짓는다. 홀드아웃 원문만 읽으므로 GPU도 `runs/` 도 필요 없다.
echo "[로컬] 페이지 짓는 중"
"$PYTHON" -m cli.pages

# 2. 서버가 코드를 따라오게 한다. `--ff-only` 는 **서버에서 뭔가 고쳤을 때 조용히
#    합치지 않고 멈추라는 뜻이다** -- 그 자리에서 알아야 나중에 헤매지 않는다.
echo
echo "[$SERVER] git pull"
ssh "$SERVER" "cd '$REMOTE' && git pull --ff-only"

# 3. 페이지를 올린다.
echo
echo "[$SERVER] 페이지 올리는 중"
ssh "$SERVER" "mkdir -p '$REMOTE/visualizations'"
scp "$PAGE" "$SERVER:$REMOTE/visualizations/"

cat <<'DONE'

올렸습니다.

  페이지만 바꿨으면   ->  브라우저 새로 고침 (F5)
  cli/*.py 도 바꿨으면 ->  서버에서 Ctrl+C 로 내리고 다시:
                            bash serve.sh
DONE
