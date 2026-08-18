#!/usr/bin/env bash
# 2라운드 스윕이 중간에 죽었을 때 이어서 돌린다. **몇 번을 다시 불러도 안전하다.**
#
# 이어서 돌리는 장치는 이미 `cli/sweep.py`의 collect()에 있다 -- `runs/<이름>/eval/
# summary.json`이 있는 실험은 큐에서 뺀다. 그래서 같은 명령을 다시 부르는 것만으로
# 끝난 것은 건너뛴다. 이 스크립트가 그 위에 더하는 것은 셋이다.
#
# 1. **두 번 띄우는 사고를 막는다.** 이게 제일 중요하다. 살아 있는데 또 부르면
#    GPU 한 장에 실험이 둘씩 물려, --slots-per-gpu 1로 피하려던 OOM이 그대로 난다.
# 2. **학습은 끝났는데 채점만 못 한 것을 구해낸다.** 어댑터(final/)는 있는데
#    summary.json이 없는 경우다. 그냥 다시 부르면 sweep이 학습부터 다시 하는데,
#    hra-r2는 그것만 11시간이다. 채점만 따로 돌려 붙여준다.
# 3. 무엇이 끝났고 무엇이 남았는지 세어 보여준다.
#
# 사용 (**저장소 뿌리에서**):
#     bash resume_sweep.sh            현황을 보고, 죽어 있으면 이어서 띄운다
#     bash resume_sweep.sh --status   현황만 보고 끝낸다
set -euo pipefail
cd "$(dirname "$0")"

ORDER="configs/r2-order.txt"
LOG="runs/r2-full.log"
HOLDOUT="data/20260811__annotate__v2.2/holdout.jsonl"
GPU="${GPU:-4}"

[ -f "$ORDER" ] || { echo "$ORDER 가 없다. git pull 부터."; exit 1; }

# CRLF(윈도우 줄끝 \r\n)를 떼고 읽는다. 이 저장소는 Windows Git으로 커밋되므로
# 줄 끝에 \r이 붙어 내려올 수 있고, 그러면 경로가 "...json\r"이 되어 파일을 못 찾는다.
mapfile -t CONFIGS < <(tr -d '\r' < "$ORDER" | grep -v '^[[:space:]]*$')

# 설정에서 name과 output_dir을 뽑는다. read_config가 상속까지 풀어준 값을 써야
# 여기서 보는 폴더와 sweep이 쓰는 폴더가 어긋나지 않는다.
PY_BIN="$(command -v python || command -v python3)" || {
    echo "python이 없다. venv를 먼저 켠다: source .venv/bin/activate"; exit 1; }

INFO=$("$PY_BIN" - "${CONFIGS[@]}" <<'PY'
import sys
from cli.train import read_config
for p in sys.argv[1:]:
    c = read_config(p)
    print(f"{p}\t{c['name']}\t{c['output_dir']}")
PY
) || { echo "설정을 못 읽었다. 위 오류를 본다."; exit 1; }

[ -n "$INFO" ] || { echo "설정에서 아무것도 안 나왔다. $ORDER 를 본다."; exit 1; }

done_n=0; orphan=(); left=()
while IFS=$'\t' read -r path name out; do
    if   [ -f "$out/eval/summary.json" ]; then done_n=$((done_n + 1))
    elif [ -d "$out/final" ];            then orphan+=("$name|$out")
    else                                      left+=("$name")
    fi
done <<< "$INFO"

echo "설정 ${#CONFIGS[@]}개 - 채점까지 끝남 $done_n · 학습만 끝남 ${#orphan[@]} · 안 됨 ${#left[@]}"
if [ ${#orphan[@]} -gt 0 ]; then printf '  학습만 끝남: %s\n' "${orphan[@]%%|*}"; fi
if [ ${#left[@]} -gt 0 ]; then
    if [ ${#left[@]} -le 10 ]; then echo "  남은 것: ${left[*]}"
    else echo "  남은 것: ${left[*]:0:10} ... 외 $(( ${#left[@]} - 10 ))개"; fi
fi

if pgrep -f "[p]ython[0-9.]* -m cli\.sweep" > /dev/null; then
    echo
    echo "**스윕이 아직 살아 있다. 아무것도 안 한다.**"
    echo "  진행: tail -f $LOG"
    echo "  정말 멈추려면: pkill -f 'run\\.(sweep|train|evaluate)'"
    exit 0
fi

if [ "${1:-}" = "--status" ]; then exit 0; fi

if [ ${#orphan[@]} -eq 0 ] && [ ${#left[@]} -eq 0 ]; then
    echo; echo "전부 끝났다. 띄울 것이 없다."; exit 0
fi

# 학습만 끝난 것은 채점만 따로 붙인다. 이걸 안 하면 sweep이 학습부터 다시 한다.
for item in "${orphan[@]+"${orphan[@]}"}"; do
    name="${item%%|*}"; out="${item#*|}"
    echo; echo "채점만 다시: $name (GPU $GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -m cli.evaluate \
        --data "$HOLDOUT" --adapter "$out/final" --out "$out/eval" \
        >> "runs/$name.log" 2>&1 \
        && echo "  붙였다" || echo "  실패. runs/$name.log 를 본다 (sweep이 학습부터 다시 한다)"
done

echo
echo "스윕을 다시 띄운다. 끝난 것은 sweep이 알아서 건너뛴다."
nohup "$PY_BIN" -m cli.sweep --slots-per-gpu 1 --configs "${CONFIGS[@]}" >> "$LOG" 2>&1 < /dev/null &
sleep 2
pgrep -f "[p]ython[0-9.]* -m cli\.sweep" > /dev/null \
    && echo "떴다 (pid $(pgrep -f '[p]ython[0-9.]* -m cli\.sweep' | head -1)). 진행: tail -f $LOG" \
    || echo "**안 떴다.** $LOG 끝을 본다."
