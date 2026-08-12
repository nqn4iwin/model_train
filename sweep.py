"""설정 여러 개를 GPU 4·5에 두 개씩 물려 돌리고, 끝나는 대로 채점한다.

한 실험은 **학습 -> 채점** 두 걸음이고, 이 둘은 같은 GPU에서 이어서 한다. 어댑터를
디스크에 쓴 뒤 다시 읽는 것보다 자리를 잡고 있는 편이 단순하다.

**두 장을 묶지 않는다.** 실험 하나가 210스텝짜리 짧은 것이라, 2장을 묶어 하나를
돌리면(DDP) 통신 비용만 붙는다. 한 장씩 두 개를 나란히 돌리는 편이 빠르다.

**`--check`를 먼저 돌린다.** 열여덟 개 중 어느 설정이 `peft`에서 아예 안 만들어지는지를
GPU를 쓰기 전에 몇 초 만에 가려낸다. IA3·VeRA처럼 구조를 가정하는 방식은 인자 이름부터
다르고, 없는 칸을 적으면 학습이 그 지점에서 멈춘다.

사용:
    python sweep.py --check                       설정만 검사하고 끝
    python sweep.py --configs configs/ia3.json configs/vera.json
    python sweep.py --all                         configs/ 전체 (_base 제외)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from train import read_config

HERE = Path(__file__).resolve().parent
HOLDOUT = "data/20260811__annotate__v2.2/holdout.jsonl"
GPUS = ["4", "5"]


def check(paths: list[Path]) -> list[Path]:
    """`peft` 설정이 실제로 만들어지는지만 본다. 모델도 GPU도 안 쓴다."""
    from peft import get_peft_config

    good = []
    for path in paths:
        config = read_config(path)
        try:
            built = get_peft_config({"task_type": "CAUSAL_LM", **config["peft"]})
            print(f"  됨    {config['name']:<24} {type(built).__name__}")
            good.append(path)
        except Exception as error:
            print(f"  안 됨  {config['name']:<24} {type(error).__name__}: {error}")
    print(f"\n  {len(good)}/{len(paths)}개가 설정 단계를 통과했습니다.")
    if len(good) < len(paths):
        print("  안 되는 것은 인자 이름이 다르거나 peft가 그 조합을 안 받는 것입니다.")
        print("  **이것도 결과입니다** -- 무엇이 KORMo에 못 붙는지가 이 스윕의 산출입니다.")
    return good


def collect(path: Path) -> dict | None:
    """이미 끝난 실험의 결과를 주워 온다. 없으면 None.

    **열여덟 시간짜리 큐는 반드시 한 번은 끊긴다** -- 터미널이 닫히거나, 한 실험이
    GPU를 못 놓거나, 사람이 멈춘다. 끝난 것을 다시 돌리지 않아야 이어서 할 수 있다.
    """
    config = read_config(path)
    summary_path = HERE / config["output_dir"] / "eval" / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"stage": "끝", "verdict": summary["verdict"],
            "note": config.get("note", ""), "peft_type": config["peft"]["peft_type"],
            **summary["AM_rates"], "평균": summary["AM_mean"],
            "교사일치": summary.get("teacher_agreement"),
            "쏠림": summary.get("skew"),
            "판정": summary.get("judgements"),
            "안 멈춤": summary["rambled_outputs"], "분": "-"}


def run_one(path: Path, gpu: str, results: dict, lock: threading.Lock) -> None:
    config = read_config(path)
    name, out = config["name"], config["output_dir"]
    log = HERE / "runs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def step(label: str, argv: list[str]) -> int:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
            handle.flush()
            return subprocess.run(
                argv, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode

    print(f"  [GPU {gpu}] 학습 시작  {name}")
    code = step("학습", ["python", "train.py", "--config", str(path)])
    if code != 0:
        with lock:
            results[name] = {"stage": "학습", "verdict": "못 돌림", "returncode": code}
        print(f"  [GPU {gpu}] 못 돌림   {name}  (로그: runs/{name}.log)")
        return

    print(f"  [GPU {gpu}] 채점 시작  {name}")
    code = step("채점", ["python", "baseline.py", "--data", HOLDOUT,
                        "--adapter", f"{out}/final", "--out", f"{out}/eval"])
    summary_path = HERE / out / "eval" / "summary.json"
    if code != 0 or not summary_path.exists():
        with lock:
            results[name] = {"stage": "채점", "verdict": "못 돌림", "returncode": code}
        print(f"  [GPU {gpu}] 채점 실패 {name}")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with lock:
        results[name] = {"stage": "끝", "verdict": summary["verdict"],
                         "note": config.get("note", ""),
                         "peft_type": config["peft"]["peft_type"],
                         **summary["AM_rates"], "평균": summary["AM_mean"],
                         "교사일치": summary.get("teacher_agreement"),
                         "쏠림": summary.get("skew"),
                         "판정": summary.get("judgements"),
                         "안 멈춤": summary["rambled_outputs"],
                         "분": round((time.time() - started) / 60, 1)}
        write_table(results)
    print(f"  [GPU {gpu}] {summary['verdict']:<6} {name}  평균 {summary['AM_mean']:.1%}")


COLUMNS = ["AM1", "AM2", "AM3", "AM6s", "AM8s", "평균", "교사일치", "쏠림",
           "판정", "안 멈춤", "분"]


def cell(value, column: str) -> str:
    """표 한 칸을 만든다. 없는 값은 `None`이 아니라 `-`로 적는다."""
    if value is None:
        return "-"
    if isinstance(value, float) and column != "분":
        return f"{value:.1%}"
    return str(value)


def write_table(results: dict) -> list[str]:
    """한 건 끝날 때마다 곧바로 표를 다시 쓴다.

    마지막에 한 번만 쓰면 **열일곱 시간째에 죽었을 때 열일곱 시간을 잃는다.**
    `annotate.py`가 697건을 한 줄씩 흘려 쓴 것과 같은 이유다.
    """
    lines = ["# 스윕 결과", "",
             "학습 전 KORMo(제로샷) 평균 7.6% · AM1 13.5% · negative 0건."
             " `docs/베이스라인_기록.md` 참고.", "",
             "**판정이 한 종류뿐이면 `붕괴`다. AM 점수를 믿지 않는다.**"
             " `labels`가 비면 AM2·AM3이 검사할 것이 없어 자동 만점이 되므로,"
             " '무조건 negative'가 이 채점기의 만점 전략이다.", "",
             "`교사일치`는 AM 항목이 아니라 붕괴를 알아보는 눈금이다."
             " 홀드아웃이 positive 26 · negative 11이라 **무조건 negative면 29.7%,"
             " 무조건 positive면 70.3%**가 나온다. 이 두 값 근처는 붕괴로 읽는다.", "",
             "`쏠림`은 제일 많이 낸 판정이 차지하는 몫이다. 100%면 붕괴고,"
             " 90%대는 붕괴에 가깝다. **합격 판정에는 안 쓰고 눈으로만 본다** --"
             " 결과를 보고 문턱을 세우면 라운드끼리 비교가 끊긴다.", "",
             "| 실험 | PEFT | 판정 | " + " | ".join(COLUMNS) + " | 메모 |",
             "| --- | --- | --- | " + " | ".join("---:" for _ in COLUMNS) + " | --- |"]
    for name, row in sorted(results.items(), key=lambda kv: -(kv[1].get("평균") or -1)):
        cells = [cell(row.get(c), c) for c in COLUMNS]
        lines.append(f"| {name} | {row.get('peft_type', '-')} | {row['verdict']} | "
                     + " | ".join(cells) + f" | {row.get('note', '')} |")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "runs" / "sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", type=Path, default=[])
    ap.add_argument("--all", action="store_true", help="configs/ 전체 (_base 제외)")
    ap.add_argument("--check", action="store_true", help="설정만 검사하고 끝낸다")
    ap.add_argument("--redo", action="store_true",
                    help="이미 끝난 실험도 다시 돌린다. 기본은 건너뛰고 결과만 주워 온다")
    ap.add_argument("--slots-per-gpu", type=int, default=2,
                    help="한 장에 동시에 물릴 실험 수. 10B를 bf16으로 올리면 21GB고 "
                         "LoRA는 원본을 얼려두므로 80GB 한 장에 둘이 들어간다. 벽시계 "
                         "시간이 절반이 되고 GPU가 노는 시간도 준다. OOM이 나면 1로 준다")
    args = ap.parse_args()

    paths = list(args.configs)
    if args.all or not paths:
        paths = sorted(p for p in (HERE / "configs").glob("*.json")
                       if not p.name.startswith("_"))

    print(f"설정 {len(paths)}개\n")
    paths = check(paths)
    if args.check or not paths:
        return

    # 이미 끝난 것은 결과만 주워 오고 큐에서 뺀다. 끊긴 큐를 이어서 하려면 이것이 필요하다.
    results: dict[str, dict] = {}
    queue = []
    for path in paths:
        done = None if args.redo else collect(path)
        if done:
            results[read_config(path)["name"]] = done
        else:
            queue.append(path)
    if results:
        print(f"\n  이미 끝난 {len(results)}개는 건너뜁니다 (--redo로 다시 돌립니다)")

    slots = [gpu for gpu in GPUS for _ in range(args.slots_per_gpu)]
    print(f"\nGPU {'·'.join(GPUS)}번에 각 {args.slots_per_gpu}개씩,"
          f" 동시에 {len(slots)}개를 돌립니다. 남은 {len(queue)}개.\n")
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                path = queue.pop(0)
            run_one(path, gpu, results, lock)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("\n".join(write_table(results)[6:]))
    print(f"\n저장: runs/sweep.md · runs/sweep.json")


if __name__ == "__main__":
    main()
