#!/usr/bin/env bash
# 2차run 을 **처음 띄운다.** 금요일 저녁에 걸어놓고 퇴근하는 용도다.
# 중간에 죽었으면 이것 말고 `run2_resume.sh` 를 쓴다.
#
# 두 걸음을 이어서 한다. **재채점이 먼저다.**
#
#   1. 재채점  prerun · 1차run 의 어댑터를 **135건 홀드아웃**으로 다시 매긴다.
#              37건 결과는 `eval/` 에 그대로 두고 `eval135/` 에 따로 쓴다 --
#              **덮어쓰면 지난 147줄 표가 사라진다.**
#   2. 학습    `configs/run2-order.txt` 의 28개를 학습 -> 채점(135건)까지.
#
# 재채점을 앞에 두는 이유는 **곡선을 그으려면 앞 두 점이 새 자로 찍혀 있어야** 하고,
# 학습이 하나라도 끝나기 전에 그 점들이 준비되기 때문이다.
#
# 사용 (**저장소 뿌리에서, venv 를 켠 뒤**):
#     bash run2_start.sh              재채점 -> 학습  (뒤로 돌린다. 바로 프롬프트가 돌아온다)
#     bash run2_start.sh --rescore    재채점만
#     bash run2_start.sh --train      학습만
#     bash run2_start.sh --fg         앞에서 돈다 (로그를 눈으로 보며 확인할 때)
set -euo pipefail
cd "$(dirname "$0")"

HOLDOUT="data/20260821__annotate__v2.2-run2A/holdout.jsonl"   # 135건. A·B 가 같다
ORDER="configs/run2-order.txt"
RESCORE_LIST="configs/run2-rescore.txt"
LOG="runs/run2.log"
GPUS=(4 5)

PY_BIN="$(command -v python || command -v python3)" || {
    echo "python이 없다. venv를 먼저 켠다: source .venv/bin/activate"; exit 1; }

STEP=""; FG=no
for a in "$@"; do
    case "$a" in
        --rescore|--train) STEP="$a" ;;
        --fg) FG=yes ;;
        *) echo "모르는 인자: $a"; exit 1 ;;
    esac
done

for f in "$HOLDOUT" "$ORDER" "$RESCORE_LIST"; do
    [ -f "$f" ] || { echo "$f 가 없다. git pull 부터."; exit 1; }
done

# ---------------------------------------------------------- 뒤로 돌리기 (자기 재호출)
# `python -u` 와 함께 쓴다. 3라운드에서 출력 버퍼 때문에 로그가 안 쌓여 죽은 줄 알았다.
if [ "${RUN2_CHILD:-}" != "1" ] && [ "$FG" = no ]; then
    if pgrep -f "[p]ython[0-9.]* -m cli\.(sweep|train|evaluate)" > /dev/null; then
        echo "**이미 뭔가 돌고 있다. 아무것도 안 한다.**"
        echo "  진행:   tail -f $LOG"
        echo "  현황:   bash check_sweep.sh $ORDER $LOG"
        echo "  이어서: bash run2_resume.sh"
        exit 0
    fi
    if [ -f runs/sweep.json ]; then
        cp -n runs/sweep.json "runs/sweep.json.bak.$(date +%Y%m%d%H%M)" 2>/dev/null || true
        echo "sweep.json 을 백업했다"
    fi
    RUN2_CHILD=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 < /dev/null &
    sleep 3
    if pgrep -f "[p]ython[0-9.]* -m cli\.(sweep|evaluate)" > /dev/null; then
        echo "떴다 (로그: $LOG)"
        echo "  진행: tail -f $LOG"
        echo "  현황: bash check_sweep.sh $ORDER $LOG"
    else
        echo "**안 떴다.** $LOG 끝을 본다."; tail -20 "$LOG" 2>/dev/null || true
    fi
    exit 0
fi

echo "===== 2차run 시작  $(date '+%F %T')  홀드아웃 $HOLDOUT"

# ---------------------------------------------------------------- 1. 재채점
rescore() {
    mapfile -t NAMES < <(tr -d '\r' < "$RESCORE_LIST" | grep -v '^[[:space:]]*$')
    echo "== 재채점 ${#NAMES[@]}개 -- 135건. 결과는 eval135/ 로 간다 (eval/ 는 안 건드린다)"
    local i=0 done_n=0 skip=0 miss=0
    for name in "${NAMES[@]}"; do
        local out="runs/$name"
        if [ -f "$out/eval135/summary.json" ]; then skip=$((skip+1)); continue; fi
        if [ ! -d "$out/final" ]; then
            echo "  **어댑터 없음** $name -- 재학습이 필요하다. 건너뛴다"; miss=$((miss+1)); continue
        fi
        local gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
        i=$(( i + 1 )); done_n=$((done_n+1))
        echo "  [GPU $gpu] $name  ($(date '+%H:%M'))"
        CUDA_VISIBLE_DEVICES="$gpu" "$PY_BIN" -u -m cli.evaluate \
            --data "$HOLDOUT" --adapter "$out/final" --out "$out/eval135" \
            >> "runs/${name}.eval135.log" 2>&1 &
        # 한 장에 하나씩만 물린다. GPU 수만큼 띄웠으면 둘 다 끝날 때까지 기다린다.
        if [ $(( i % ${#GPUS[@]} )) -eq 0 ]; then wait; fi
    done
    wait
    echo "== 재채점 끝 -- 돌린 것 $done_n · 이미 있던 것 $skip · 어댑터 없음 $miss"
}

# ---------------------------------------------------------------- 2. 학습
train() {
    mapfile -t CONFIGS < <(tr -d '\r' < "$ORDER" | grep -v '^[[:space:]]*$')
    echo "== 학습 ${#CONFIGS[@]}개 -- 긴 것부터. 끝난 것은 sweep 이 알아서 건너뛴다"
    "$PY_BIN" -u -m cli.sweep --slots-per-gpu 1 --holdout "$HOLDOUT" \
        --configs "${CONFIGS[@]}"
}

case "$STEP" in
    --rescore) rescore ;;
    --train)   train ;;
    *)         rescore; train ;;
esac

echo "===== 전부 끝났다  $(date '+%F %T')"
