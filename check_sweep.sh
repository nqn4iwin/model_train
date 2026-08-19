#!/usr/bin/env bash
# 스윕이 **정상으로 끝났는지 죽었는지**를 가려 본다. 아무것도 안 띄우고 안 고친다.
#
# `resume_sweep.sh`와 헷갈리지 않는다. 저쪽은 **이어서 띄우는** 것이고 이쪽은
# **읽기만** 한다. 프로세스가 내려간 것을 보고 제일 먼저 알아야 하는 것은
# "이어서 돌릴까"가 아니라 **"끝나서 내려간 건가, 죽어서 내려간 건가"**다.
# 다 끝나서 내려간 것을 이어 돌리면 아무 일도 안 일어나지만, 죽은 줄 모르고
# 결과를 표에 넣으면 **없는 실험을 있는 것으로 읽는다.**
#
# 사용 (**저장소 뿌리에서**):
#     bash check_sweep.sh                          3라운드(기본)를 본다
#     bash check_sweep.sh configs/r2-order.txt     다른 라운드를 본다
#     bash check_sweep.sh configs/r3-order.txt runs/r3-lr.log
#
# 두 번째 인자는 **스윕 자신의 표준출력 로그**다. 실험별 로그(`runs/<이름>.log`)와
# 다르다 -- 이쪽은 `nohup ... > 여기`로 받은 파일이고, 큐 전체의 시작과 끝이 찍힌다.
set -euo pipefail
cd "$(dirname "$0")"

ORDER="${1:-configs/r3-order.txt}"
STDOUT_LOG="${2:-runs/r3-lr.log}"

[ -f "$ORDER" ] || { echo "$ORDER 가 없다."; exit 1; }

# CRLF(윈도우 줄끝 \r\n)를 뗀다. 이 저장소는 Windows Git으로 커밋되므로 줄 끝에 \r이
# 붙어 내려오고, 그러면 경로가 "...json\r"이 되어 파일을 못 찾는다.
mapfile -t CONFIGS < <(tr -d '\r' < "$ORDER" | grep -v '^[[:space:]]*$')

PY_BIN="$(command -v python || command -v python3)" || {
    echo "python이 없다. venv를 먼저 켠다: source .venv/bin/activate"; exit 1; }

# 설정 상속(`extends`)을 푼 뒤의 값을 써야, 여기서 보는 폴더와 스윕이 쓴 폴더가 같다.
INFO=$("$PY_BIN" - "${CONFIGS[@]}" <<'PY'
import sys
from cli.train import read_config
for p in sys.argv[1:]:
    c = read_config(p)
    print(f"{c['name']}\t{c['output_dir']}")
PY
) || { echo "설정을 못 읽었다. venv가 켜져 있는지 본다."; exit 1; }

echo "== $ORDER -- 설정 ${#CONFIGS[@]}개"
echo

# ---------------------------------------------------------------- 1. 살아 있나
ALIVE=no
if pgrep -f "[p]ython[0-9.]* -m cli\.sweep" > /dev/null; then
    ALIVE=yes
    echo "스윕 프로세스   **살아 있다** (pid $(pgrep -f '[p]ython[0-9.]* -m cli\.sweep' | tr '\n' ' '))"
else
    echo "스윕 프로세스   내려갔다"
fi

# 자식(학습·채점)이 남아 있는지도 본다. 부모만 죽고 자식이 GPU를 쥔 채 남는 일이 있다.
CHILD=$(pgrep -f "[p]ython[0-9.]* -m cli\.(train|evaluate)" 2>/dev/null | wc -l)
[ "${CHILD:-0}" -gt 0 ] && echo "학습·채점 자식   ${CHILD}개가 아직 돌고 있다 (부모만 죽었을 수 있다)"

# ------------------------------------------------- 2. 표준출력에 끝 표시가 있나
#
# `cli.sweep`은 큐를 다 비운 뒤에만 마지막 줄로 `저장: runs/sweep.md ...`를 찍는다.
# **그 줄이 있으면 스스로 끝난 것이다.** 없는데 프로세스도 없으면 중간에 끊긴 것이다.
#
# 다만 없다고 곧바로 죽었다고 못 박지 않는다. 파이썬은 터미널이 아닌 곳으로 출력할 때
# 버퍼에 모아 뒀다 한꺼번에 쓰는데, **정상 종료 때는 그 버퍼를 반드시 비우고 나간다.**
# 그래서 끝 표시가 없다는 것은 사실상 "정상으로 안 나갔다"와 같다 -- 다만 로그 파일을
# 잘못 짚었을 때도 똑같이 안 보이므로, 아래에서 실험 폴더를 따로 센다.
FINISHED=no
if [ -f "$STDOUT_LOG" ]; then
    if grep -q "저장: runs/sweep.md" "$STDOUT_LOG"; then
        FINISHED=yes
        echo "$STDOUT_LOG   끝 표시 있음 ($(wc -l < "$STDOUT_LOG")줄)"
    else
        echo "$STDOUT_LOG   **끝 표시 없음** ($(wc -l < "$STDOUT_LOG")줄, 마지막 기록 $(date -r "$STDOUT_LOG" '+%m-%d %H:%M'))"
    fi
else
    echo "$STDOUT_LOG   파일이 없다 (로그 이름을 두 번째 인자로 준다)"
fi
echo

# ------------------------------------------------------------ 3. 실험별 상태표
#
# `summary.json`이 있으면 채점까지 끝난 것이고, `final/`만 있으면 학습만 끝난 것이다.
# `resume_sweep.sh`가 이어 돌릴 때 보는 것과 **같은 기준**이다.
printf '%-24s %-10s %-12s %s\n' 이름 상태 "마지막 기록" "실험 로그 마지막 줄"
done_n=0; train_n=0; none_n=0; SUSPECT=""
while IFS=$'\t' read -r name out; do
    log="runs/$name.log"
    if   [ -f "$out/eval/summary.json" ]; then state="끝";      done_n=$((done_n+1))
    elif [ -d "$out/final" ];            then state="학습만";   train_n=$((train_n+1))
    elif [ -e "$log" ] || [ -d "$out" ]; then state="**끊김**"; none_n=$((none_n+1)); SUSPECT="$name"
    else                                      state="안 시작";  none_n=$((none_n+1))
    fi

    when="-"; last="-"
    if [ -f "$log" ]; then
        when=$(date -r "$log" '+%m-%d %H:%M')
        # 빈 줄과 `=====` 구분선은 건너뛰고 내용이 있는 마지막 줄을 집는다.
        last=$(grep -v '^[[:space:]]*$' "$log" | grep -v '^=*$' | tail -1 | cut -c1-60)
    fi
    printf '%-24s %-10s %-12s %s\n' "$name" "$state" "$when" "$last"
done <<< "$INFO"

echo
echo "채점까지 끝남 $done_n · 학습만 끝남 $train_n · 못 감 $none_n"

# ---------------------------------------------- 4. 못 간 것이 있으면 왜인지 본다
#
# 실험 로그에서 **원인이 되는 줄만** 뽑는다. 로그 하나가 10만 줄이라 통째로 보면 못 읽는다.
if [ -n "$SUSPECT" ] && [ -f "runs/$SUSPECT.log" ]; then
    echo
    echo "== 못 간 것 중 마지막: $SUSPECT -- 원인 후보"
    grep -nE "OutOfMemoryError|CUDA out of memory|Traceback|KeyboardInterrupt|Killed|No module|FileNotFoundError" \
        "runs/$SUSPECT.log" | tail -5 || echo "  (원인 줄이 안 잡힌다. tail -50 runs/$SUSPECT.log 를 직접 본다)"
fi

# --------------------------------------------------------------------- 5. 판정
#
# **끝 표시가 판정의 주인이다.** 실험 몇 개가 실패한 것과 스윕이 죽은 것은 다른 일이다 --
# 2라운드는 66개 중 4개가 OOM으로 못 돌았지만 스윕 자신은 큐를 끝까지 비우고 정상으로
# 나갔다. 그 넷은 자리(80GB 한 장)에 안 들어가는 것이라 이어 돌려도 똑같이 실패한다.
# 그래서 "$done_n 이 전부인가"로 판정하면 **정상 종료를 사고로 잘못 읽는다.**
echo
FAILED=$(( ${#CONFIGS[@]} - done_n - train_n ))
if [ "$ALIVE" = yes ]; then
    echo "판정: **아직 돌고 있다.** 아무것도 하지 않는다. 진행은 tail -f $STDOUT_LOG"
    echo "      실험별 진행은 runs/<이름>.log 나 runs/sweep.md 가 더 빨리 쌓인다."
    exit 0
fi

if [ "$FINISHED" = yes ]; then
    echo "판정: **정상 종료다.** 스윕이 큐를 끝까지 비우고 스스로 나갔다."
    if [ "$FAILED" -gt 0 ] || [ "$train_n" -gt 0 ]; then
        echo "      다만 ${#CONFIGS[@]}개 중 채점까지 간 것은 $done_n개다."
        echo "      **못 간 것은 스윕이 죽어서가 아니라 그 실험이 실패한 것이다.**"
        echo "      위 「원인 후보」를 본다. OOM이면 이어 돌려도 같은 자리에서 또 실패한다."
    fi
elif [ "$done_n" -eq "${#CONFIGS[@]}" ]; then
    echo "판정: **끝난 것으로 보인다.** ${#CONFIGS[@]}개 전부 채점까지 끝났다."
    echo "      다만 $STDOUT_LOG 에 끝 표시가 없다 -- 로그 파일을 잘못 짚었거나,"
    echo "      마지막 표를 쓰기 직전에 죽었을 수 있다. 산출물은 다 있으므로 결과는 쓴다."
else
    echo "판정: **비정상 종료다.** 큐를 다 비우기 전에 끊겼다."
    echo "      ${#CONFIGS[@]}개 중 채점까지 $done_n개 · 학습만 $train_n개 · 못 감 $FAILED개."
    echo
    echo "**이어 돌리기 전에 표를 뜬다:  cp runs/sweep.json runs/sweep.json.bak**"
    echo "  cli.sweep 은 끝날 때 sweep.json 을 이번에 넘긴 설정만으로 덮는다."
    echo
    echo "이어서 돌리려면 (채점까지 끝난 것은 cli.sweep 이 알아서 건너뛴다):"
    echo "      nohup python -u -m cli.sweep --slots-per-gpu 1 \\"
    echo "          --configs \$(tr -d '\\r' < $ORDER) >> $STDOUT_LOG 2>&1 &"
    echo "  **resume_sweep.sh 는 쓰지 않는다** -- 저것은 2라운드 전용이다"
    echo "  (configs/r2-order.txt 가 안에 박혀 있어 인자를 안 받는다)."
    echo "  python **-u** 를 붙인다 -- 없으면 출력이 버퍼에 갇혀 로그가 한참 안 쌓인다."
    echo "  **학습만 끝난 것이 있으면** 그것부터 채점만 따로 붙인다. 안 그러면"
    echo "  cli.sweep 이 학습부터 다시 한다 (hra-r2 는 그것만 41시간이었다):"
    echo "      CUDA_VISIBLE_DEVICES=4 python -m cli.evaluate \\"
    echo "          --data data/20260811__annotate__v2.2/holdout.jsonl \\"
    echo "          --adapter runs/<이름>/final --out runs/<이름>/eval"
fi

echo
echo "결과를 표에 넣는 것은 **로컬에서** 한다:  bash sync_runs.sh  ->  python -m cli.rescore --write"
echo "**서버의 runs/sweep.md · sweep.json 은 당겨오지 않는다** -- 이번 라운드 몇 줄짜리 토막이라"
echo "로컬의 140줄을 덮는다. sync_runs.sh 가 이미 그 둘을 빼고 받는다."
