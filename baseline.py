"""학습 전 KORMo를 채점한다. **이 값이 없으면 학습 후 점수를 해석할 수 없다.**

    학습 전이 20%였는데 학습 후 65%  ->  학습이 먹었다
    학습 전이 90%였는데 학습 후 65%  ->  학습이 망쳤다

같은 65%인데 정반대다. 기획서 3.3이 '베이스라인: 제로샷 KORMo'로 적어둔 값이 이것이다.

**학습 후 채점에도 같은 파일을 쓴다.** `--adapter`에 LoRA 어댑터 경로를 주면 그것을
얹어 같은 홀드아웃에 같은 잣대를 건다. 재는 도구가 갈라지면 앞뒤를 비교할 수 없다.

**생성은 기본이 greedy다.** 온도를 주면 같은 조건을 다시 돌렸을 때 값이 흔들려,
학습 효과와 노이즈를 못 가른다. 흔들림을 보고 싶으면 `--temperature`와 `--repeat`을
같이 준다.

사용:
    python baseline.py --data data/20260811__annotate__v2.2/holdout.jsonl \\
                       --out runs/baseline-kormo-rules
    python baseline.py ... --adapter runs/lora-full-rules/final
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from formatting import build_prompt
from scoring import KEYS, restatement_ratio, score_blind, verdict

MODEL = "KORMo-Team/KORMo-10B-base"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="build_records.py가 낸 JSONL")
    ap.add_argument("--out", required=True, help="결과를 쓸 폴더")
    ap.add_argument("--no-rules", action="store_true",
                    help="역할 A 규칙서를 안 붙인다. 제로샷에는 붙이는 것이 기본이다 -- "
                         "학습이 없으면 대상 7종·방향 5종이라는 어휘 자체를 모른다")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-revision", default="main",
                    help="가중치 판본. 어느 가중치로 잰 값인지 기록에 남는다")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 어댑터 경로. 주면 학습 후 값이 된다")
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="JSON 하나에 넉넉한 길이. 모자라면 끝이 잘려 파싱이 깨진다")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0이면 greedy. 0이 아니면 --repeat과 같이 쓴다")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만. 0이면 전체")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in
            Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # KORMo는 자체 모델 클래스(KORMoForCausalLM)를 쓰는 것으로 보인다. 표준 아키텍처가
    # 아니면 trust_remote_code 없이는 로드 자체가 안 된다. 여기서 터지면 그것이 곧
    # PEFT 방식들이 이 모델에 안 붙는 이유가 되므로, 실패도 기록할 값이다.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda:0")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = (out_dir / "records.jsonl").open("w", encoding="utf-8")

    started = time.time()
    graded: list[dict] = []
    for turn in range(1, args.repeat + 1):
        for index, row in enumerate(rows, 1):
            prompt = build_prompt(row, rules=not args.no_rules)
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id)
            # 프롬프트를 되풀이한 부분은 잘라낸다. 안 자르면 프롬프트 안의 예시 JSON이
            # 모델 답으로 읽혀 파싱률이 실제보다 높게 나온다.
            raw = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)

            marked = score_blind(raw)
            parsed = marked.pop("parsed") or {}
            record = {
                "id": row["id"], "turn": turn, "scores": marked,
                "judgement": str(parsed.get("judgement", "")).strip(),
                "labels": parsed.get("labels"), "impacts": parsed.get("impacts"),
                "direct_impact": parsed.get("direct_impact"),
                "restatement_ratio": restatement_ratio(
                    row["after"], str(parsed.get("direct_impact") or "")),
                "raw": raw,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            graded.append(record)
            print(f"  [{turn}.{index}/{len(rows)}] {sum(marked.values())}/5  "
                  f"{record['judgement'] or '?':<9} {row['id']}")
    stream.close()

    rates = {k: round(sum(r["scores"][k] for r in graded) / len(graded), 3) for k in KEYS}
    summary = {
        "model": args.model, "model_revision": args.model_revision,
        "adapter": args.adapter, "data": args.data, "rules": not args.no_rules,
        "items": len(rows), "repeat": args.repeat, "calls": len(graded),
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "AM_rates": rates,
        "AM_mean": round(sum(rates.values()) / len(rates), 3),
        "AM_min": min(rates.values()),
        "verdict": verdict(rates),
        "judgements": {j: sum(1 for r in graded if r["judgement"] == j)
                       for j in {r["judgement"] for r in graded}},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    for key, rate in rates.items():
        print(f"  {key:<6} {rate:>6.1%}")
    print(f"  {'평균':<6} {summary['AM_mean']:>6.1%}   최저 {summary['AM_min']:>6.1%}")
    print(f"\n  판정: {summary['verdict']}   ({summary['elapsed_seconds']:.0f}초)")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
