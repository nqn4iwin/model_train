"""설정 JSON 하나로 KORMo에 PEFT 학습을 한 번 돌린다.

`continuous-training/train.py`에서 골격을 가져와 이 프로젝트에 맞게 고친 것이다.
바뀐 곳은 셋이다.

1. **GPU 2장 강제를 뗐다.** 원본은 2장이 아니면 실행을 거부한다. 여기서 돌릴 실험은
   짧은 것 스무 개라, 2장을 묶어 하나를 돌리는 것(DDP)보다 **한 장씩 두 개를 나란히
   돌리는 편이 빠르다** -- 짧은 실험에 DDP는 통신 비용만 붙는다. 대신 아무것도
   지정하지 않고 0번을 잡는 사고는 막는다. 이 서버는 8장을 여럿이 나눠 쓴다.
2. **데이터를 우리 JSONL에서 읽는다.** 프롬프트와 정답은 `sft/formatting.py`가 조립한다.
   조건(규칙서를 붙일지, 정답을 어디까지 둘지)이 데이터가 아니라 설정에 있어야
   "다른 건 똑같이 두고 하나만 바꾼다"가 성립한다.
3. **PEFT 방식을 설정에서 고른다.** `peft.peft_type`에 이름을 적으면 그대로 만든다.
   LoRA뿐 아니라 IA3·VeRA 같은 것도 이름만 바꿔 넣을 수 있고, **KORMo에 안 붙는
   방식은 여기서 에러가 난다.** 그것을 알아내는 것이 이 스윕의 목적이다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    python -m cli.train --config configs/delora.json --inspect   # 안 돌리고 확인만
    CUDA_VISIBLE_DEVICES=4 python -m cli.train --config configs/delora.json
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sft.formatting import build_completion, build_prompt
from sft.models import is_qwen, load_causal_lm, load_tokenizer
from sft.records import ruler_name
from sft.training import ROOT, git_revision, load_rows, read_config, to_dataset

def inspect(config: dict, rows: list[dict]) -> None:
    """안 돌리고 한 건이 실제로 어떻게 토큰이 되는지 보여준다.

    **EOS가 제일 위험하다.** 학습 전 KORMo는 답을 다 쓰고도 멈추지 않는다(baseline에서
    37건 전부 그랬다). 정답 끝에 종료 토큰이 안 붙으면 학습 후에도 안 멈추고, 그러면
    뒤에 붙은 딴소리 때문에 AM1이 영영 안 오른다. **돌리기 전에 눈으로 본다.**
    """
    tokenizer = load_tokenizer(config["model"], revision=config.get("model_revision", "main"))
    row = rows[0]
    prompt = build_prompt(row, rules=config.get("rules", True))
    completion = build_completion(row, config.get("target", "full"))

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    print(f"레코드 {len(rows)}건 · 첫 건 {row['id']}")
    print(f"  프롬프트 {len(prompt_ids):>5}토큰   (규칙서 {'붙임' if config.get('rules', True) else '안 붙임'})")
    print(f"  정답     {len(completion_ids):>5}토큰   (target={config.get('target', 'full')})")
    print(f"  합계     {len(prompt_ids) + len(completion_ids):>5}토큰"
          f"   / max_length {config.get('max_length', 4096)}")

    over = sum(1 for r in rows
               if len(tokenizer(build_prompt(r, rules=config.get("rules", True)),
                                add_special_tokens=False)["input_ids"])
               + len(tokenizer(build_completion(r, config.get("target", "full")),
                               add_special_tokens=False)["input_ids"])
               > config.get("max_length", 4096))
    print(f"  max_length를 넘는 레코드: {over}건"
          f"{'  <- 잘려서 정답 끝이 사라집니다' if over else ''}")

    print(f"\n  eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  정답 마지막 5토큰: {completion_ids[-5:]}"
          f" -> {tokenizer.convert_ids_to_tokens(completion_ids[-5:])}")
    if tokenizer.eos_token_id in completion_ids[-3:]:
        print("  EOS가 정답 끝에 있습니다.")
    else:
        print("  ! 정답 문자열 자체에는 EOS가 없습니다. TRL이 붙여 주는지"
              " 아래 실제 배치에서 확인하세요.")

    # TRL을 실제로 세워 본다. **`Adding EOS to train dataset` 로그가 뜨면 그것이 답이다**
    # -- 정답 문자열에 EOS가 없어도 TRL이 붙여 준다는 뜻이고, 2026-08-11에 확인했다.
    # 그 아래 배치를 꺼내는 부분은 TRL 5.x에서 실패할 수 있는데, 위 로그가 이미
    # 물음에 답했으므로 실패해도 상관없다.
    try:
        import torch
        from peft import get_peft_config
        from trl import SFTConfig, SFTTrainer
        # KORMo는 trust_remote_code가 있어야 로드된다. Qwen은 dtype 없이 불렀다가는
        # 27B를 fp32로 CPU에 올리려다 죽는다 -- bf16을 명시한다.
        model_arg = config["model"]
        model_init_kwargs = {"trust_remote_code": True}
        if is_qwen(config["model"]):
            model_arg = _inspect_qwen(config)
            model_init_kwargs = None
        trainer = SFTTrainer(
            model=model_arg,
            args=SFTConfig(output_dir="/tmp/inspect", max_length=config.get("max_length", 4096),
                           report_to=[], model_init_kwargs=model_init_kwargs),
            train_dataset=to_dataset(rows[:2], config),
            processing_class=tokenizer,
            peft_config=get_peft_config({"task_type": "CAUSAL_LM", **config["peft"]}))
        if is_qwen(config["model"]):
            trainable = sum(parameter.numel() for parameter in trainer.model.parameters()
                            if parameter.requires_grad)
            if not trainable:
                raise RuntimeError("Qwen LoRA에 학습되는 parameter가 없습니다.")
            print(f"  LoRA 학습 parameter: {trainable:,}개")
        batch = trainer.train_dataset[0]
        ids, labels = batch["input_ids"], batch.get("labels", [])
        print(f"\n  TRL이 만든 배치: input_ids {len(ids)}개 · labels {len(labels)}개")
        print(f"    마지막 5토큰 {ids[-5:]} -> {tokenizer.convert_ids_to_tokens(ids[-5:])}")
        if labels:
            masked = sum(1 for x in labels if x == -100)
            print(f"    손실에서 제외된 토큰 {masked}개 (프롬프트 길이 {len(prompt_ids)}와 비슷해야 정상)")
        print(f"    끝에 EOS({tokenizer.eos_token_id})가 있나: {ids[-1] == tokenizer.eos_token_id}")
    except Exception as error:
        print(f"\n  배치를 꺼내지는 못했습니다: {type(error).__name__}: {error}")
        if is_qwen(config["model"]):
            raise
        print("  위에 'Adding EOS to train dataset'이 찍혔으면 EOS는 붙은 것입니다.")


def _inspect_qwen(config: dict):
    """Qwen 전용 진단. 실제로 한 번 불러 세 가지를 확인한다.

    이 저장소가 이미 겪은 위험 -- `target_modules`에 없는 이름의 층은 에러 없이
    조용히 학습에서 빠진다 -- 를 학습을 실제로 돌리기 전에 걸러내려는 것이다. Qwen은
    선형 주의(GatedDeltaNet) 층과 전체 주의 층의 Linear 이름이 서로 달라 이 위험이
    KORMo보다 크다.
    """
    import torch
    from torch import nn

    print("\n  Qwen 진단 (실제로 한 번 불러 확인합니다) ...")
    model = load_causal_lm(config["model"], revision=config.get("model_revision", "main"),
                           dtype=torch.bfloat16)
    class_name = type(model).__name__
    ok = "" if class_name == "Qwen3_5ForCausalLM" else "  <- 텍스트 전용 클래스가 아닙니다. 로딩 경로를 확인하세요"
    print(f"  불러온 클래스: {class_name}{ok}")

    vision_like = [name for name, _ in model.named_children()
                   if "vis" in name.lower() or "image" in name.lower()]
    print(f"  vision 관련 서브모듈: {vision_like or '없음 (정상)'}")

    target_modules = set(config["peft"].get("target_modules", []))
    linear_leaf_names = {name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                         if isinstance(module, nn.Linear)}
    matched = target_modules & linear_leaf_names
    print(f"  target_modules 중 실제로 걸리는 이름: {sorted(matched)}"
          f" ({len(matched)}/{len(target_modules)})")
    if not matched:
        print("  ! 하나도 안 걸립니다 -- 이 target_modules로는 LoRA가 통째로 안 붙습니다.")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--inspect", action="store_true",
                    help="학습하지 않고 한 건이 어떻게 토큰이 되는지만 보여준다")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="학습 step 수 상한. GPU smoke test에는 1을 준다")
    ap.add_argument("--output-dir", default=None,
                    help="결과 폴더 override. smoke test를 본 실험과 분리할 때 쓴다")
    args = ap.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        ap.error("--max-steps는 1 이상이어야 합니다")
    config = read_config(args.config)
    rows = load_rows(config)
    max_steps = args.max_steps if args.max_steps is not None else config.get("max_steps", -1)

    if args.inspect:
        inspect(config, rows)
        return

    # 이 서버는 8장을 여럿이 나눠 쓴다. 안 지정하면 0번을 잡는데 우리 몫은 4·5번이다.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES를 지정하세요. 이 프로젝트 몫은 4·5번입니다.\n"
            "  CUDA_VISIBLE_DEVICES=4 python train.py --config ...")

    import torch
    from peft import get_peft_config
    from trl import SFTConfig, SFTTrainer

    output_name = args.output_dir or config["output_dir"]
    output_dir = ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**config, "output_dir": output_name, "git_revision": git_revision(),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "rows": len(rows), "max_steps": max_steps}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    tokenizer = load_tokenizer(config["model"], revision=config.get("model_revision", "main"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 설정에 적힌 이름 그대로 만든다. KORMo에 안 붙는 방식은 여기서 터지고, **그 실패도
    # 기록할 값이다** -- 무엇이 이 모델에 붙고 안 붙는지가 이 스윕이 그리려는 지도다.
    peft_config = get_peft_config({"task_type": "CAUSAL_LM", **config["peft"]})

    # 어디에 기록할지. **기본값은 텐서보드 하나뿐이다** -- `wandb`를 기본으로 두면 그것이
    # 안 깔린 환경에서 학습이 시작조차 못 한다. 설정에 적은 곳만 켠다.
    report_to = config.get("report_to", ["tensorboard"])
    if "wandb" in report_to:
        try:
            import wandb  # noqa: F401
        except ImportError:
            # **터뜨리지 않고 빼되, 조용히 빼지도 않는다.** 설정에 적어 놓고 기록이 안
            # 남는 것이 제일 나쁘다 -- 학습이 다 끝난 뒤에야 알아챈다.
            print("! wandb를 못 불러왔습니다. 텐서보드만 씁니다."
                  "  (pip install wandb)")
            report_to = [r for r in report_to if r != "wandb"]
        else:
            # 프로젝트 이름을 코드에서 정해 둔다. 셸에 `WANDB_PROJECT`를 안 걸어두면
            # 실험이 엉뚱한 곳에 쌓이는데, 나중에야 알아채는 종류의 사고다.
            # setdefault라 셸에서 지정한 값이 있으면 그쪽이 이긴다.
            os.environ.setdefault("WANDB_PROJECT",
                                  config.get("wandb_project", "model_train"))

    # KORMo는 문자열 그대로 TRL에 넘겨 안에서 불러오게 둔다(지금까지 그래 왔다).
    # Qwen은 TRL의 문자열 기반 로딩이 architectures가 가리키는 멀티모달 클래스로
    # 안 풀릴 위험이 있어 확인이 안 됐으므로, 직접 불러 객체를 넘긴다 -- 이미
    # 실체화된 객체라 model_init_kwargs는 못 쓴다(TRL이 재로딩하지 않는다).
    model_arg = config["model"]
    model_init_kwargs = {"dtype": torch.bfloat16, "use_cache": False,
                         "trust_remote_code": True}
    if is_qwen(config["model"]):
        model_arg = load_causal_lm(config["model"], revision=config.get("model_revision", "main"),
                                   dtype=torch.bfloat16, use_cache=False)
        model_init_kwargs = None

    trainer = SFTTrainer(
        model=model_arg,
        args=SFTConfig(
            output_dir=str(output_dir),
            seed=config.get("seed", 42),
            max_length=config.get("max_length", 4096),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            learning_rate=config.get("learning_rate", 2e-4),
            num_train_epochs=config.get("num_train_epochs", 3),
            max_steps=max_steps,
            logging_steps=config.get("logging_steps", 5),
            save_strategy=config.get("save_strategy", "epoch"),
            save_total_limit=config.get("save_total_limit", 1),
            bf16=True,
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            model_init_kwargs=model_init_kwargs,
            # 기록에 붙는 이름. 안 주면 output_dir(`runs/delora`)가 이름이 되어
            # W&B 목록에서 전부 `runs/`로 시작한다. 설정 이름이 곧 실험 이름이다.
            run_name=config["name"],
            report_to=report_to),
        train_dataset=to_dataset(rows, config),
        processing_class=tokenizer,
        peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()
    print(f"\n저장: {output_dir / 'final'}")
    print("채점:")
    print(f"  CUDA_VISIBLE_DEVICES=$GPU python -m cli.evaluate \\")
    print(f"      --data data/20260811__annotate__v2.2/holdout.jsonl \\")
    # **채점 폴더 이름은 홀드아웃이 정한다**(`sft.records.ruler_name`) -- 37건이면
    # `eval-mof`, 135건이면 `eval-mof-motie`다. 이 실험이 쓴 데이터 폴더 옆에 있는
    # 홀드아웃이 그 실험의 자이므로 거기서 뽑는다. 없으면 자리표시자를 적는다.
    beside = ROOT / Path(config["data"]).with_name("holdout.jsonl")
    folder = ruler_name(beside) if beside.exists() else "<채점폴더>"
    print(f"      --adapter {output_name}/final"
          f" --out {output_name}/{folder}")


if __name__ == "__main__":
    main()
