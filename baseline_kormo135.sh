#!/usr/bin/env bash
# 「학습 안 함 · KORMo」 칸을 135건 자로 채운다. **6칸 비교에서 이 한 칸만 비어 있다.**
#
#            학습 안 함        조건 A          조건 B
#   KORMo    <- 이 스크립트    delora-run2A    delora-run2B
#   Qwen     baseline-qwen     qwen-delora-run2A  qwen-delora-run2B
#
# 왜 다시 재나. `runs/baseline-kormo-rules/` 에 이미 학습 전 KORMo 값이 있지만
# **37건(mof) 자로 잰 것**이고 폴더 구조도 `eval-` 하위 폴더가 생기기 전의 평평한
# 모양이다. 135건 자에서는 기준선 자체가 달라 그 값을 옮겨 쓸 수 없다 --
# "무조건 positive"가 37건에서는 70.3%인데 135건에서는 83.7%다.
# 이 스크립트는 그 옛 파일을 **건드리지 않는다.** 옆에 새 폴더를 만든다.
#
# 조건은 Qwen 기준선과 한 칸도 안 다르게 맞췄다. 전부 `cli/evaluate.py` 의 기본값이라
# 인자로 적을 것이 없다 -- 규칙서 붙임 · prefill 붙임 · max_new_tokens 768 ·
# temperature 0 · few-shot 예시는 `20260821__annotate__v2.2-run2A/train.jsonl` 에서.
# **예시 고르기가 난수를 안 쓰므로**(`sft.formatting.select_fewshot`) Qwen 이 본 것과
# 글자 그대로 같은 3건이 들어간다. 그래야 두 모델의 3-shot 값을 나란히 놓을 수 있다.
#
# 먼저 venv 를 켠다. 안 켜면 시스템 파이썬이 잡혀 torch 를 못 찾는다.
#     cd /data1/yblee/repository/model_train && source .venv/bin/activate
#     bash baseline_kormo135.sh
#
# 6·7번 GPU 에 하나씩 물려 나란히 돌린다. 한 건에 한 시간 안팎이다
# (학습한 KORMo 가 41분이고, 기준선은 말을 안 멈춰 더 걸린다).
set -euo pipefail

HOLDOUT="${HOLDOUT:-data/20260821__annotate__v2.2-run2A/holdout.jsonl}"
RUN="${RUN:-runs/baseline-kormo-rules}"

# `python -u` 로 띄운다. 안 붙이면 출력이 버퍼에 갇혀 **로그가 한참 안 쌓이고**,
# 2026-08-18 에 그것을 보고 죽은 줄 알았다.
CUDA_VISIBLE_DEVICES=6 python -u -m cli.evaluate \
  --data "$HOLDOUT" --out "$RUN/eval-mof-motie" \
  > runs/kormo-eval-0shot.log 2>&1 &
zero=$!

CUDA_VISIBLE_DEVICES=7 python -u -m cli.evaluate \
  --data "$HOLDOUT" --shots 3 --out "$RUN/eval-mof-motie-3shot" \
  > runs/kormo-eval-3shot.log 2>&1 &
three=$!

# `wait` 하나가 실패해도 나머지를 기다린다. `set -e` 아래에서 그냥 `wait` 를 쓰면
# 먼저 죽은 쪽에서 스크립트가 끊겨 **다른 쪽이 살았는지 죽었는지 안 남는다.**
fail=0
wait "$zero"  || { echo "0-shot 이 실패했습니다. runs/kormo-eval-0shot.log 를 보세요"; fail=1; }
wait "$three" || { echo "3-shot 이 실패했습니다. runs/kormo-eval-3shot.log 를 보세요"; fail=1; }
[ "$fail" -eq 0 ] || exit 1

echo
echo "끝났습니다. 로컬에서 당겨 표에 넣으려면:"
echo "  bash sync_runs.sh"
echo "  python -m cli.rescore --data $HOLDOUT --write"
