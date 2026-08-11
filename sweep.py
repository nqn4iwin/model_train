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
                         **summary["AM_rates"],
                         "평균": summary["AM_mean"], "최저": summary["AM_min"],
                         "안 멈춤": summary["rambled_outputs"],
                         "분": round((time.time() - started) / 60, 1)}
    print(f"  [GPU {gpu}] {summary['verdict']:<6} {name}  평균 {summary['AM_mean']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", type=Path, default=[])
    ap.add_argument("--all", action="store_true", help="configs/ 전체 (_base 제외)")
    ap.add_argument("--check", action="store_true", help="설정만 검사하고 끝낸다")
    args = ap.parse_args()

    paths = list(args.configs)
    if args.all or not paths:
        paths = sorted(p for p in (HERE / "configs").glob("*.json")
                       if not p.name.startswith("_"))

    print(f"설정 {len(paths)}개\n")
    paths = check(paths)
    if args.check or not paths:
        return

    # GPU 두 자리를 큐로 두고, 자리가 나는 대로 다음 것을 물린다.
    print(f"\nGPU {'·'.join(GPUS)}번에 두 개씩 물립니다.\n")
    results: dict[str, dict] = {}
    lock = threading.Lock()
    queue = list(paths)

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                path = queue.pop(0)
            run_one(path, gpu, results, lock)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 표로 남긴다. 스무 개를 로그에서 눈으로 줍는 것은 못 할 일이다.
    columns = ["AM1", "AM2", "AM3", "AM6s", "AM8s", "평균", "최저", "안 멈춤", "분"]
    lines = ["# 스윕 결과", "",
             "학습 전 KORMo(제로샷) 평균 7.6% · AM1 13.5%. `docs/베이스라인_기록.md` 참고.", "",
             "| 실험 | PEFT | 판정 | " + " | ".join(columns) + " | 메모 |",
             "| --- | --- | --- | " + " | ".join("---:" for _ in columns) + " | --- |"]
    for name, row in sorted(results.items(),
                            key=lambda kv: -(kv[1].get("평균") or -1)):
        cells = [f"{row[c]:.1%}" if isinstance(row.get(c), float) and c not in ("분",)
                 else str(row.get(c, "-")) for c in columns]
        lines.append(f"| {name} | {row.get('peft_type', '-')} | {row['verdict']} | "
                     + " | ".join(cells) + f" | {row.get('note', '')} |")
    (HERE / "runs" / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "runs" / "sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]))
    print(f"\n저장: runs/sweep.md · runs/sweep.json")


if __name__ == "__main__":
    main()
