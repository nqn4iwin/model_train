#!/usr/bin/env bash
# GPU 를 반납하기 전 마지막 밤에 돌리는 것. **두 가지를 두 장에 갈라 붙인다.**
#
#   4번   delora-sentence-s44-run2A 학습 + 채점   -> 축 1 이 seed 3 대 3 이 된다
#   5번   프리필을 학습 모양으로 맞춰 다시 채점    -> 학습이 필요 없다. 어댑터만 쓴다
#
# ── 5번이 무엇을 묻나 ───────────────────────────────────────────────
# `sft.formatting.build_completion` 이 학습 정답으로 주는 것은 **한 줄짜리 압축 JSON**
# 이다.
#
#     {"judgement": "positive", "direct_impact": "…
#
# 그런데 `cli/evaluate.py` 의 `--prefill` 기본값은 **여러 줄 들여쓴 모양**으로 답을
# 시작시킨다.
#
#     {\n  "judgement": "
#
# **모델을 학습 때 한 번도 못 본 모양으로 출발시키는 것이다.** 2026-08-26 에 확인한
# delora-sentence-s43-run2A 의 깨진 25건이 전부 이 여러 줄 모양으로 시작해 문장 도중에
# 멈춰 있다(토큰 상한과 무관하다 — 잘림 0건).
#
# **`sentence` 만 다시 재면 안 된다.** 그러면 프리필과 조건을 한꺼번에 바꾸는 것이라
# 좋아져도 무엇 덕인지 못 가른다. `full` 을 대조로 같이 잰다 — **`full` 은 그대로인데
# `sentence` 만 좋아져야 프리필이 범인이다.**
#
# ── 기존 값은 안 건드린다 ───────────────────────────────────────────
# 결과를 `eval-mof-motie-prefill1/` 이라는 **다른 폴더**에 쓴다. 표에는
# `<실험>-prefill1` 이라는 별도 줄로 앉는다(`cli/rescore.py` 가 자 이름 뒤의 꼬리를
# 줄 이름에 붙인다). **`docs/TODO.md` 가 경고하는 "바꾸면 학습 전 값과 비교가 끊긴다"
# 는 덮어쓸 때의 이야기이고, 옆에 새로 쓰면 안 끊긴다.**
#
# 사용:
#     cd /data1/yblee/repository/model_train && source .venv/bin/activate
#     nohup bash lastnight.sh > runs/lastnight.log 2>&1 &
set -euo pipefail

HOLDOUT="data/20260821__annotate__v2.2-run2A/holdout.jsonl"
# 학습 정답과 글자 그대로 같은 시작. `build_completion` 의 json.dumps 기본 구분자다.
PREFILL='{"judgement": "'

# ── 4번: s44 sentence ────────────────────────────────────────────
gpu4() {
  # **`export` 를 함수 안에 둔다.** `CUDA_VISIBLE_DEVICES=4 gpu4` 처럼 함수 이름 앞에
  # 붙이면 bash 에서 그 값이 자식에게 갈지가 미묘하다. 이 함수는 `&` 로 띄워 서브셸에서
  # 돌므로 여기서 export 해도 바깥 셸로 새지 않는다.
  export CUDA_VISIBLE_DEVICES=4
  local name=delora-sentence-s44-run2A
  python -u -m cli.train --config "configs/$name.json" || return 1
  python -u -m cli.evaluate --data "$HOLDOUT" \
    --adapter "runs/$name/final" --out "runs/$name/eval-mof-motie"
}

# ── 5번: 프리필 재채점 넷 ─────────────────────────────────────────
# sentence 둘 + full 둘. **full 이 대조군이라 빼면 안 된다.**
gpu5() {
  export CUDA_VISIBLE_DEVICES=5
  local name
  for name in delora-sentence-run2A delora-sentence-s43-run2A \
              delora-run2A delora-s43-run2A; do
    python -u -m cli.evaluate --data "$HOLDOUT" \
      --adapter "runs/$name/final" --prefill "$PREFILL" \
      --out "runs/$name/eval-mof-motie-prefill1" \
      || echo "  ! $name 프리필 재채점 실패 — 다음으로 넘어갑니다"
  done
}

gpu4 > runs/lastnight-gpu4.log 2>&1 &
four=$!
gpu5 > runs/lastnight-gpu5.log 2>&1 &
five=$!

# `set -e` 아래에서 그냥 `wait` 를 쓰면 먼저 죽은 쪽에서 스크립트가 끊겨 **다른 쪽이
# 살았는지 죽었는지 안 남는다.** 둘 다 기다린 뒤에 판정한다.
fail=0
wait "$four" || { echo "4번(s44) 실패 — runs/lastnight-gpu4.log"; fail=1; }
wait "$five" || { echo "5번(프리필) 실패 — runs/lastnight-gpu5.log"; fail=1; }

echo
echo "끝났습니다. 로컬에서:"
echo "  bash sync_runs.sh"
echo "  python -m cli.rescore --data $HOLDOUT --write"
[ "$fail" -eq 0 ] || exit 1
