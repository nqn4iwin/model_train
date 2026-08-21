#!/usr/bin/env bash
# 2차run 이 **중간에 죽었을 때 이어서 돌린다.** 주말에 들여다보고 쓰는 용도다.
# **몇 번을 다시 불러도 안전하다.**
#
# `run2_start.sh` 와 헷갈리지 않는다. 저쪽은 **처음 띄우는** 것이고 이쪽은
# **끊긴 자리부터 잇는** 것이다. 이어 도는 장치 자체는 양쪽에 다 있다 --
# 재채점은 `eval135/summary.json` 이 있으면 건너뛰고, 학습은 `cli/sweep.py` 가
# `eval/summary.json` 이 있으면 큐에서 뺀다. 이 스크립트가 그 위에 더하는 것은 셋이다.
#
#   1. **두 번 띄우는 사고를 막는다.** 살아 있는데 또 부르면 GPU 한 장에 실험이
#      둘씩 물려, --slots-per-gpu 1 로 피하려던 OOM 이 그대로 난다.
#   2. **학습은 끝났는데 채점만 못 한 것을 구해낸다.** 어댑터(final/)는 있는데
#      eval/summary.json 이 없는 경우다. 그냥 다시 부르면 sweep 이 학습부터
#      다시 하는데, DeLoRA 한 건이 3,178건에서 약 2시간이다.
#   3. 무엇이 끝났고 무엇이 남았는지 세어 보여준다.
#
# 사용 (**저장소 뿌리에서, venv 를 켠 뒤**):
#     bash run2_resume.sh            현황을 보고, 죽어 있으면 이어서 띄운다
#     bash run2_resume.sh --status   현황만 보고 끝낸다
set -euo pipefail
cd "$(dirname "$0")"

HOLDOUT="data/20260821__annotate__v2.2-run2A/holdout.jsonl"
ORDER="configs/run2-order.txt"
RESCORE_LIST="configs/run2-rescore.txt"
LOG="runs/run2.log"
GPU="${GPU:-4}"

PY_BIN="$(command -v python || command -v python3)" || {
    echo "python이 없다. venv를 먼저 켠다: source .venv/bin/activate"; exit 1; }
for f in "$HOLDOUT" "$ORDER" "$RESCORE_LIST"; do
    [ -f "$f" ] || { echo "$f 가 없다. git pull 부터."; exit 1; }
done

# ------------------------------------------------------------------ 1. 재채점 현황
# CRLF(윈도우 줄끝 \r\n)를 뗀다. 이 저장소는 Windows Git 으로 커밋되므로 줄 끝에
# \r 이 붙어 내려오고, 그러면 이름이 "...-r2\r" 이 되어 폴더를 못 찾는다.
mapfile -t NAMES < <(tr -d '\r' < "$RESCORE_LIST" | grep -v '^[[:space:]]*$')
rs_done=0; rs_left=(); rs_miss=()
for n in "${NAMES[@]}"; do
    if   [ -f "runs/$n/eval135/summary.json" ]; then rs_done=$((rs_done+1))
    elif [ ! -d "runs/$n/final" ];              then rs_miss+=("$n")
    else                                             rs_left+=("$n"); fi
done
echo "재채점  ${#NAMES[@]}개 - 끝남 $rs_done · 남음 ${#rs_left[@]} · 어댑터없음 ${#rs_miss[@]}"
[ ${#rs_miss[@]} -gt 0 ] && echo "  어댑터 없음: ${rs_miss[*]}"

# ------------------------------------------------------------------ 2. 학습 현황
mapfile -t CONFIGS < <(tr -d '\r' < "$ORDER" | grep -v '^[[:space:]]*$')
# 설정 상속(extends)을 푼 뒤의 값을 써야, 여기서 보는 폴더와 sweep 이 쓰는 폴더가 같다.
INFO=$("$PY_BIN" - "${CONFIGS[@]}" <<'PY'
import sys
from cli.train import read_config
for p in sys.argv[1:]:
    c = read_config(p)
    print(f"{p}\t{c['name']}\t{c['output_dir']}")
PY
) || { echo "설정을 못 읽었다. venv 가 켜져 있는지 본다."; exit 1; }

tr_done=0; orphan=(); tr_left=()
while IFS=$'\t' read -r path name out; do
    if   [ -f "$out/eval/summary.json" ]; then tr_done=$((tr_done+1))
    elif [ -d "$out/final" ];            then orphan+=("$name|$out")
    else                                       tr_left+=("$name"); fi
done <<< "$INFO"
echo "학습    ${#CONFIGS[@]}개 - 채점까지 끝남 $tr_done · 학습만 끝남 ${#orphan[@]} · 안 됨 ${#tr_left[@]}"
[ ${#orphan[@]} -gt 0 ] && printf '  학습만 끝남: %s\n' "${orphan[@]%%|*}"
if [ ${#tr_left[@]} -gt 0 ]; then
    if [ ${#tr_left[@]} -le 8 ]; then echo "  남은 것: ${tr_left[*]}"
    else echo "  남은 것: ${tr_left[*]:0:8} ... 외 $(( ${#tr_left[@]} - 8 ))개"; fi
fi

# ------------------------------------------------------------------ 3. 살아 있나
if pgrep -f "[p]ython[0-9.]* -m cli\.(sweep|train|evaluate)" > /dev/null; then
    echo
    echo "**아직 살아 있다. 아무것도 안 한다.**"
    echo "  진행: tail -f $LOG"
    echo "  정말 멈추려면: pkill -f 'cli\\.(sweep|train|evaluate)'"
    exit 0
fi

[ "${1:-}" = "--status" ] && exit 0

if [ $rs_done -eq $(( ${#NAMES[@]} - ${#rs_miss[@]} )) ] \
   && [ ${#orphan[@]} -eq 0 ] && [ ${#tr_left[@]} -eq 0 ]; then
    echo; echo "전부 끝났다. 띄울 것이 없다."; exit 0
fi

# 학습만 끝난 것은 채점만 따로 붙인다. 이걸 안 하면 sweep 이 학습부터 다시 한다.
for item in "${orphan[@]+"${orphan[@]}"}"; do
    name="${item%%|*}"; out="${item#*|}"
    echo; echo "채점만 다시: $name (GPU $GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u -m cli.evaluate \
        --data "$HOLDOUT" --adapter "$out/final" --out "$out/eval" \
        >> "runs/$name.log" 2>&1 \
        && echo "  붙였다" || echo "  실패. runs/$name.log 를 본다 (sweep 이 학습부터 다시 한다)"
done

echo
echo "이어서 띄운다. 끝난 것은 알아서 건너뛴다."
bash run2_start.sh
