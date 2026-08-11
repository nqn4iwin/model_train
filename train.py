"""설정 JSON 하나로 KORMo에 PEFT 학습을 한 번 돌린다.

`continuous-training/train.py`에서 골격을 가져와 이 프로젝트에 맞게 고친 것이다.
바뀐 곳은 셋이다.

1. **GPU 2장 강제를 뗐다.** 원본은 2장이 아니면 실행을 거부한다. 여기서 돌릴 실험은
   짧은 것 스무 개라, 2장을 묶어 하나를 돌리는 것(DDP)보다 **한 장씩 두 개를 나란히
   돌리는 편이 빠르다** -- 짧은 실험에 DDP는 통신 비용만 붙는다. 대신 아무것도
   지정하지 않고 0번을 잡는 사고는 막는다. 이 서버는 8장을 여럿이 나눠 쓴다.
2. **데이터를 우리 JSONL에서 읽는다.** 프롬프트와 정답은 `formatting.py`가 조립한다.
   조건(규칙서를 붙일지, 정답을 어디까지 둘지)이 데이터가 아니라 설정에 있어야
   "다른 건 똑같이 두고 하나만 바꾼다"가 성립한다.
3. **PEFT 방식을 설정에서 고른다.** `peft.peft_type`에 이름을 적으면 그대로 만든다.
   LoRA뿐 아니라 IA3·VeRA 같은 것도 이름만 바꿔 넣을 수 있고, **KORMo에 안 붙는
   방식은 여기서 에러가 난다.** 그것을 알아내는 것이 이 스윕의 목적이다.

사용:
    python train.py --config configs/lora-full-rules.json --inspect   # 안 돌리고 확인만
    CUDA_VISIBLE_DEVICES=4 python train.py --config configs/lora-full-rules.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from formatting import build_completion, build_prompt

HERE = Path(__file__).resolve().parent


def read_config(path: str | Path) -> dict:
    """설정을 읽는다. `extends`가 있으면 그 파일 위에 이 파일을 덮어쓴다.

    **비교 실험이 전부 "다른 건 똑같이 두고 하나만 바꾼다"라 상속이 필요하다.** 스무 개
    설정에 학습률을 스무 번 적어두면, 한 곳을 고칠 때 열아홉 곳이 옛 값으로 남아
    통제가 조용히 깨진다. 바뀌는 칸만 적으면 **파일이 곧 그 실험의 차이 목록**이 된다.

    한 단계만 물려받는다. 상속의 상속은 무엇이 이겼는지 읽기 어려워진다.
    """
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        base = json.loads((path.parent / parent).read_text(encoding="utf-8"))
        base.pop("extends", None)
        # **output_dir은 절대 물려받지 않는다.** 물려받으면 열여덟 실험이 한 폴더를
        # 가리켜 서로를 덮어쓴다. 에러가 안 나고 결과만 사라지는 종류의 고장이라
        # 나중에 알아채기도 어렵다. 자식이 명시하지 않았으면 이름에서 새로 만든다.
        base.pop("output_dir", None)
        config = {**base, **config}
    config.setdefault("output_dir", f"runs/{config['name']}")
    missing = {"name", "model", "data", "peft"} - set(config)
    if missing:
        raise ValueError(f"{path}: 설정에 빠진 칸 {sorted(missing)}")
    return config


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def load_rows(config: dict) -> list[dict]:
    """학습 레코드를 읽고 설정이 시키는 대로 거른다.

    **거르기를 학습 시점에 하는 이유**는 데이터 파일을 한 벌로 얼려두기 위해서다.
    negative를 빼는 조건과 안 빼는 조건이 같은 파일을 쓰면, 두 실험이 정말 같은
    데이터에서 출발했는지가 파일 해시 하나로 확인된다.
    """
    rows = [json.loads(line) for line in
            (HERE / config["data"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    if config.get("negatives", "keep") == "drop":
        rows = [r for r in rows if r["judgement"] != "negative"]

    # 한 계열이 전체의 60%라 그대로 두면 그 문체만 배운다. 계열 안에서는 문서 전체에
    # 퍼지도록 고른다 -- 앞에서 자르면 문서 앞머리의 표제부·날짜만 남는다.
    limit = config.get("downsample")
    if limit:
        by_series: dict[str, list[dict]] = {}
        for row in rows:
            by_series.setdefault(row["series"], []).append(row)
        rows = [group[(i * len(group)) // min(limit, len(group))]
                for group in by_series.values()
                for i in range(min(limit, len(group)))]
    return rows


def to_dataset(rows: list[dict], config: dict):
    """prompt와 completion 두 칸짜리로 만든다.

    TRL은 이 두 칸을 보면 **프롬프트 부분의 손실을 빼고 정답 부분만 학습한다.**
    한 덩어리 텍스트로 주면 규칙서 3,300자까지 외우게 되는데, 그것은 우리가 가르치려는
    것이 아니다.
    """
    from datasets import Dataset

    return Dataset.from_list([
        {"prompt": build_prompt(row, rules=config.get("rules", True)),
         "completion": build_completion(row, config.get("target", "full"))}
        for row in rows])


def inspect(config: dict, rows: list[dict]) -> None:
    """안 돌리고 한 건이 실제로 어떻게 토큰이 되는지 보여준다.

    **EOS가 제일 위험하다.** 학습 전 KORMo는 답을 다 쓰고도 멈추지 않는다(baseline에서
    37건 전부 그랬다). 정답 끝에 종료 토큰이 안 붙으면 학습 후에도 안 멈추고, 그러면
    뒤에 붙은 딴소리 때문에 AM1이 영영 안 오른다. **돌리기 전에 눈으로 본다.**
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config.get("model_revision", "main"),
        trust_remote_code=True)
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
        from trl import SFTConfig, SFTTrainer
        trainer = SFTTrainer(
            model=config["model"],
            args=SFTConfig(output_dir="/tmp/inspect", max_length=config.get("max_length", 4096),
                           report_to=[], model_init_kwargs={"trust_remote_code": True}),
            train_dataset=to_dataset(rows[:2], config),
            processing_class=tokenizer)
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
        print("  위에 'Adding EOS to train dataset'이 찍혔으면 EOS는 붙은 것입니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--inspect", action="store_true",
                    help="학습하지 않고 한 건이 어떻게 토큰이 되는지만 보여준다")
    args = ap.parse_args()
    config = read_config(args.config)
    rows = load_rows(config)

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
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = HERE / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**config, "git_revision": git_revision(),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "rows": len(rows)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config.get("model_revision", "main"),
        trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 설정에 적힌 이름 그대로 만든다. KORMo에 안 붙는 방식은 여기서 터지고, **그 실패도
    # 기록할 값이다** -- 무엇이 이 모델에 붙고 안 붙는지가 이 스윕이 그리려는 지도다.
    peft_config = get_peft_config({"task_type": "CAUSAL_LM", **config["peft"]})

    trainer = SFTTrainer(
        model=config["model"],
        args=SFTConfig(
            output_dir=str(output_dir),
            seed=config.get("seed", 42),
            max_length=config.get("max_length", 4096),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            learning_rate=config.get("learning_rate", 2e-4),
            num_train_epochs=config.get("num_train_epochs", 3),
            logging_steps=config.get("logging_steps", 5),
            save_strategy=config.get("save_strategy", "epoch"),
            save_total_limit=config.get("save_total_limit", 1),
            bf16=True,
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            model_init_kwargs={"dtype": torch.bfloat16, "use_cache": False,
                               "trust_remote_code": True},
            report_to=["tensorboard"]),
        train_dataset=to_dataset(rows, config),
        processing_class=tokenizer,
        peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()
    print(f"\n저장: {output_dir / 'final'}")
    print("채점:")
    print(f"  CUDA_VISIBLE_DEVICES=$GPU python baseline.py \\")
    print(f"      --data data/20260811__annotate__v2.2/holdout.jsonl \\")
    print(f"      --adapter {config['output_dir']}/final --out {config['output_dir']}/eval")


if __name__ == "__main__":
    main()
