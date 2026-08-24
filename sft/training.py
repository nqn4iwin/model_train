"""학습 설정과 데이터 변환처럼 CLI 여러 곳이 함께 쓰는 기능."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sft.formatting import build_completion, build_prompt

ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict:
    """한 단계 `extends`를 합친 실험 설정을 읽는다."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        base = json.loads((path.parent / parent).read_text(encoding="utf-8"))
        base.pop("extends", None)
        base.pop("output_dir", None)
        config = {**base, **config}
    config.setdefault("output_dir", f"runs/{config['name']}")
    missing = {"name", "model", "data", "peft"} - set(config)
    if missing:
        raise ValueError(f"{path}: 설정에 빠진 칸 {sorted(missing)}")
    return config


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() or None


def load_rows(config: dict) -> list[dict]:
    """설정의 negative·계열별 downsample 조건을 적용한 학습 행을 읽는다."""
    rows = [json.loads(line) for line in (ROOT / config["data"]).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if config.get("negatives", "keep") == "drop":
        rows = [row for row in rows if row["judgement"] != "negative"]
    limit = config.get("downsample")
    if limit:
        by_series: dict[str, list[dict]] = {}
        for row in rows:
            by_series.setdefault(row["series"], []).append(row)
        rows = [group[(index * len(group)) // min(limit, len(group))]
                for group in by_series.values()
                for index in range(min(limit, len(group)))]
    return rows


def to_dataset(rows: list[dict], config: dict):
    from datasets import Dataset

    return Dataset.from_list([
        {"prompt": build_prompt(row, rules=config.get("rules", True)),
         "completion": build_completion(row, config.get("target", "full"))}
        for row in rows])
