"""공유 학습 설정이 Qwen 자식 설정을 제대로 합치는지 확인한다."""
from pathlib import Path
import unittest

from sft.training import read_config

ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTest(unittest.TestCase):
    def test_qwen_child_config_keeps_its_own_run_directory(self) -> None:
        config = read_config(ROOT / "configs/qwen-delora-run2A.json")
        self.assertEqual(config["name"], "qwen-delora-run2A")
        self.assertEqual(config["output_dir"], "runs/qwen-delora-run2A")
        self.assertEqual(config["model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(config["peft"]["peft_type"], "DELORA")


if __name__ == "__main__":
    unittest.main()
