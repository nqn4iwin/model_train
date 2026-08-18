"""KORMo를 홀드아웃에 걸어 채점한다. **학습 전과 후에 같은 파일을 쓴다.**

`--adapter` 없이 부르면 학습 전(제로샷) 값이고, 주면 그 어댑터를 얹은 학습 후 값이다.
**이 둘이 같은 도구로 재져야 한다.**

    학습 전이 20%였는데 학습 후 65%  ->  학습이 먹었다
    학습 전이 90%였는데 학습 후 65%  ->  학습이 망쳤다

같은 65%인데 정반대다. 기획서 3.3이 '베이스라인: 제로샷 KORMo'로 적어둔 값이 앞엣것이고,
`docs/베이스라인_기록.md`에 있다.

2026-08-12에 `baseline.py`에서 이름을 바꿨다. 학습 전 값을 재려고 만들었지만 실제로는
학습 후 채점에 훨씬 많이 쓰여, 이름이 하는 일과 어긋나 있었다.

**생성은 기본이 greedy다.** 온도를 주면 같은 조건을 다시 돌렸을 때 값이 흔들려,
학습 효과와 노이즈를 못 가른다. 흔들림을 보고 싶으면 `--temperature`와 `--repeat`을
같이 준다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    CUDA_VISIBLE_DEVICES=4 python -m cli.evaluate \\
        --data data/20260811__annotate__v2.2/holdout.jsonl \\
        --out runs/baseline-kormo-rules
    CUDA_VISIBLE_DEVICES=4 python -m cli.evaluate ... --adapter runs/delora/final
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sft.formatting import build_prompt
from sft.scoring import (KEYS, collapsed, label_agreement, restatement_ratio,
                         score_blind, skew, verdict)

MODEL = "KORMo-Team/KORMo-10B-base"


def first_object_closed(raw: str) -> bool:
    """첫 JSON 객체가 온전히 닫혔는지 괄호를 세어 본다. **채점에 쓰지 않는다.**

    base 모델은 종료를 몰라 답을 다 쓰고도 계속 이어 쓴다. 그래서 생성이 언제나
    상한에 닿고, 상한에 닿았다는 것만으로는 **답이 잘린 것인지 뒤에 딴소리가 붙은
    것인지 가릴 수 없다.** 첫 객체가 닫혔으면 답은 다 나온 것이므로 상한을 늘려도
    소용이 없다 -- 그 경우까지 "늘리세요"라고 하면 사람을 엉뚱한 데로 보낸다.
    """
    depth = 0
    for char in raw:
        depth += (char == "{") - (char == "}")
        if depth == 0 and char == "}":
            return True
    return False


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
    ap.add_argument("--prefill", default='{\n  "judgement": "',
                    help="프롬프트 끝에 붙여 답을 시작시키는 실마리. base 모델은 규칙서를 "
                         "그냥 문서로 읽고 바로 끝내버린다. 채점 전에 출력 앞에 다시 "
                         "이어 붙이므로 JSON은 온전한 채로 매겨진다. 빈 문자열이면 안 붙인다")
    args = ap.parse_args()

    # 이 서버는 8장을 여럿이 나눠 쓴다. CUDA_VISIBLE_DEVICES를 안 주면 0번을 잡는데,
    # 0번은 우리 몫이 아니다(4·5번이다). 남의 실험 위에 얹히면 양쪽이 다 죽는다.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES를 지정하세요. 이 프로젝트 몫은 4·5번입니다.\n"
            "  CUDA_VISIBLE_DEVICES=4 python -m cli.evaluate ...")

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
            prompt = build_prompt(row, rules=not args.no_rules) + args.prefill
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id)
            # 프롬프트를 되풀이한 부분은 잘라낸다. 안 자르면 프롬프트 안의 예시 JSON이
            # 모델 답으로 읽혀 파싱률이 실제보다 높게 나온다.
            fresh = generated[0][encoded["input_ids"].shape[1]:]
            # 실마리는 우리가 넣어준 것이지만 답의 일부다. 다시 이어 붙여야 JSON이
            # 여는 중괄호를 갖춘 온전한 모양이 되어 AM1이 제대로 매겨진다.
            raw = args.prefill + tokenizer.decode(fresh, skip_special_tokens=True)

            marked = score_blind(raw)
            parsed = marked.pop("parsed") or {}
            record = {
                "id": row["id"], "turn": turn, "scores": marked,
                # 교사가 이 블록을 뭐라 판정했는지. **AM 점수에는 안 들어간다** --
                # 교사는 사람이 아니라 정답키가 못 되고, 잣대를 섞으면 교사 값과의
                # 비교가 끊긴다. 붕괴를 알아보는 데만 쓴다.
                "teacher_judgement": row.get("judgement"),
                # 교사가 읽어낸 (대상, 방향)도 같이 남긴다. 이게 없으면 라벨일치를
                # 나중에 다시 매길 때 홀드아웃을 따로 열어 id로 맞붙여야 한다
                # (`cli/rescore.py`가 옛 실험에 대해 그렇게 한다).
                "teacher_labels": row.get("labels"),
                # 0이면 모델이 곧바로 끝냈다는 뜻이고, max_new_tokens와 같으면 끝이
                # 잘렸다는 뜻이다. 0점의 원인이 이 둘 중 어느 쪽인지 여기서 갈린다.
                "new_tokens": int(fresh.shape[0]),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "first_object_closed": first_object_closed(raw),
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
                  f"{record['judgement'] or '?':<9} {record['new_tokens']:>4}토큰"
                  f"  {row['id']}")
    stream.close()

    rates = {k: round(sum(r["scores"][k] for r in graded) / len(graded), 3) for k in KEYS}
    said = [r["judgement"] for r in graded]
    # 교사와 판정이 얼마나 맞았나. **AM 항목이 아니다** -- 교사는 사람 정답키가 아니라
    # 합격 판정에 쓸 수 없고, 여기서는 붕괴를 알아보는 눈금으로만 쓴다. 홀드아웃이
    # positive 26 · negative 11이므로 무조건 negative를 내면 29.7%가 나온다.
    matched = sum(1 for r in graded if r["judgement"] == r.get("teacher_judgement"))
    is_collapsed = collapsed(said)
    # `sentence` 조건은 라벨을 구조적으로 안 낸다. **출력만 보고 정하면 붕괴와
    # 구별이 안 되므로** 실제로 돌아간 설정에서 읽는다(`cli/train.py`가 남긴다).
    ran_config = out_dir.parent / "config.json"
    target = (json.loads(ran_config.read_text(encoding="utf-8")).get("target")
              if ran_config.exists() else None)
    labels = label_agreement([(r.get("labels"), r.get("teacher_labels")) for r in graded],
                             label_free=(target == "sentence"))
    summary = {
        "model": args.model, "model_revision": args.model_revision,
        "adapter": args.adapter, "data": args.data, "rules": not args.no_rules,
        "items": len(rows), "repeat": args.repeat, "calls": len(graded),
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "prefill": args.prefill,
        "empty_outputs": sum(1 for r in graded if r["new_tokens"] == 0),
        # 상한에 닿았는데 첫 객체도 안 닫힌 것만 진짜 잘린 것이다. 닫혔는데 상한에
        # 닿은 것은 답 뒤에 딴소리가 붙은 것이라 상한을 늘려도 달라지지 않는다.
        "truncated_outputs": sum(1 for r in graded
                                 if r["new_tokens"] >= args.max_new_tokens
                                 and not r["first_object_closed"]),
        "rambled_outputs": sum(1 for r in graded
                               if r["new_tokens"] >= args.max_new_tokens
                               and r["first_object_closed"]),
        "AM_rates": rates,
        "AM_mean": round(sum(rates.values()) / len(rates), 3),
        "AM_min": min(rates.values()),
        "verdict": "붕괴" if is_collapsed else verdict(rates),
        "collapsed": is_collapsed,
        # 붕괴는 예·아니오라 "37건 중 35건이 negative" 같은 것을 못 잡는다. 판정에는
        # 안 쓰고 숫자만 남긴다 -- 결과를 보고 문턱을 세우면 비교가 끊긴다.
        "skew": skew(said),
        "teacher_agreement": round(matched / len(graded), 3),
        # 판정 한 칸이 아니라 **그 안의 내용**이 교사와 같은가. 교사일치가 두 라운드
        # 똑같이 91.9%인데 이 값은 19.2% -> 34.6%로 움직인 적이 있다. 형식만 보는
        # 눈금으로는 안 보이던 자리다(`docs/2차_스윕_결과.md` 7절).
        # **verdict에는 안 쓴다** -- 문턱을 건드리면 라운드끼리 비교가 끊긴다.
        "label_agreement": labels["rate"],
        "label_matched": labels["matched"],
        "label_denominator": labels["denominator"],
        "label_note": labels["note"],
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
    print(f"\n  판정 분포 {summary['judgements']}"
          f"   교사와 일치 {summary['teacher_agreement']:.1%}")
    if summary["label_agreement"] is None:
        print(f"  라벨일치 해당 없음 ({summary['label_note']})"
              "   -- 0%가 아닙니다. 다른 조건과 나란히 놓지 마십시오")
    else:
        print(f"  라벨일치 {summary['label_agreement']:.1%}"
              f" ({summary['label_matched']}/{summary['label_denominator']})"
              "   교사가 라벨을 단 건만 셉니다")
    if is_collapsed:
        print(f"  ! 붕괴 -- {len(graded)}건 전부 '{said[0]}'입니다. AM 점수는 믿을 수 없습니다.")
        print("    labels가 비면 AM2·AM3이 검사할 것이 없어 자동 만점이 됩니다.")
    if summary["empty_outputs"]:
        print(f"\n  ! 아무것도 안 뱉은 것 {summary['empty_outputs']}건 -- 점수가 아니라 고장입니다")
    if summary["truncated_outputs"]:
        print(f"\n  ! 답이 잘린 것 {summary['truncated_outputs']}건 -- --max-new-tokens를 늘리세요")
    if summary["rambled_outputs"]:
        print(f"\n  · 답을 다 쓰고도 안 멈춘 것 {summary['rambled_outputs']}건"
              f" -- 상한을 늘려도 달라지지 않습니다 (base 모델이라 종료를 모릅니다)")
    print(f"\n  판정: {summary['verdict']}   ({summary['elapsed_seconds']:.0f}초)")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
