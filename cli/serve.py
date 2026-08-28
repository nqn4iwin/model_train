"""정리 페이지의 「실습」 칸이 물어보는 추론 서버. **GPU를 쓰는 유일한 쪽이다.**

여섯 칸을 띄우지만 **GPU에 올라가는 것은 세 벌**이다. 어댑터(원본 가중치를 얼려두고
작은 층만 학습한 것)는 원본에 얹는 것이라, 같은 원본을 쓰는 어댑터끼리는 **한 벌에
여러 개를 붙여 두고 갈아 끼울 수 있다.**

    GPU-A  KORMo-10B + DeLoRA 어댑터 1개    ~21GB   학습 안 됨 · 잘된 KORMo
    GPU-A  KORMo-10B + LoRA   어댑터 3개    ~21GB   이상함 셋
    GPU-B  Qwen3.8-27B + DeLoRA 어댑터 1개  ~56GB   잘된 Qwen

**KORMo를 두 벌 올리는 이유는 peft가 한 모델에 DELORA와 LORA를 섞어 못 붙이기
때문이다.** 같은 종류끼리는 한 벌로 되므로 LoRA 셋은 묶었다. 「학습 안 됨」은 원본을
따로 싣지 않고 `disable_adapter()`로 어댑터를 잠깐 꺼서 만든다 -- 어댑터를 끄면 남는
것이 정확히 학습 전 KORMo다.

**답을 만드는 길은 `cli/evaluate.py`와 한 글자도 다르지 않다.** 같은 `build_prompt`,
같은 프리필, 같은 greedy, 같은 `parse_output`이다. 다르면 실습에서 나온 답이 페이지에
적힌 숫자와 어긋나고, 어긋나는 순간 둘 중 어느 쪽이 맞는지 아무도 모르게 된다.

**새 패키지를 안 쓴다.** 파이썬에 딸려 오는 `http.server`로 짰다. 이 저장소는 판본을
올리지 않기로 되어 있고(`docs/서버환경.md`), 웹 서버 하나 때문에 그 금을 넘을 이유가
없다.

사용 (**저장소 뿌리에서 `-m`으로 부른다**):
    CUDA_VISIBLE_DEVICES=4,5 python -m cli.serve --port 8000
    python -m cli.serve --list                     # 안 싣고 목록만 본다
    CUDA_VISIBLE_DEVICES=4 python -m cli.serve --only kormo-base kormo-good

브라우저가 이 서버에 닿게 하려면 로컬에서 굴을 판다. 켜 둔 채로 페이지를 연다.
    ssh -L 8000:localhost:8000 <서버주소>
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sft.formatting import build_prompt
from sft.models import load_causal_lm, load_tokenizer
from sft.scoring import parse_output

ROOT = Path(__file__).resolve().parents[1]

KORMO = "KORMo-Team/KORMo-10B-base"
QWEN = "Qwen/Qwen3.8-27B"

# `cli/evaluate.py`의 기본값 그대로다. base 모델은 규칙서를 그냥 문서로 읽고 바로
# 끝내버려서, 답의 첫 글자를 우리가 넣어 줘야 이어 쓴다.
PREFILL = '{\n  "judgement": "'

# 한 벌에 여러 어댑터를 붙여 두는 묶음. `engine`이 같으면 같은 원본 위에서 이름만
# 갈아 끼운다. `adapter`가 None이면 어댑터를 꺼서 학습 전 모델이 된다.
#
# 여기 적힌 순서가 페이지에 서는 순서다. 학습 전 -> 잘된 것 -> 이상한 것 순으로
# 두었다. 이상한 셋을 뒤에 모은 것은 **앞의 둘을 본 뒤라야 무엇이 이상한지 보이기**
# 때문이다.
CATALOG = [
    {
        "key": "kormo-base", "engine": "kormo-delora", "adapter": None,
        "label": "학습 안 됨 · KORMo",
        "run": None,
        "note": "KORMo-10B 원본. 어댑터를 안 붙였다",
        "detail": "규칙서를 지시가 아니라 읽을거리로 받아, 프롬프트 안의 예시를 "
                  "되풀이하거나 조문을 이어 쓴다. 홀드아웃 135건에서 판정이 읽힌 것이 "
                  "81건뿐이고 그중 negative는 0건이다",
    },
    {
        "key": "kormo-good", "engine": "kormo-delora", "adapter": "delora-run2A",
        "label": "잘 학습됨 · KORMo",
        "run": "delora-run2A",
        "note": "DeLoRA · 학습 3,178건 · 3에폭",
        "detail": "교사일치 94.1% · 라벨일치 37.2%. 두 라운드 다 seed 폭이 가장 "
                  "안 흔들린 계열이라 A/B를 가르는 자로 쓰던 조합이다",
    },
    {
        "key": "qwen-good", "engine": "qwen", "adapter": "qwen-delora-run2A",
        "label": "잘 학습됨 · Qwen",
        "run": "qwen-delora-run2A",
        "note": "DeLoRA · 같은 데이터 · 같은 seed · Qwen3.8-27B",
        "detail": "교사일치 94.1% · 라벨일치 55.8%. KORMo 쪽과 방법·데이터·seed가 "
                  "같아 **모델 크기만 다르다**가 성립하는 짝이다. 학습에 18시간 38분이 "
                  "걸렸다 -- 같은 스텝에서 KORMo(1시간 46분)의 10.5배다",
    },
    {
        "key": "kormo-bad-negative", "engine": "kormo-lora",
        "adapter": "lora-sentence-bare-r2",
        "label": "이상함 ① 겉만 멀쩡한 붕괴",
        "run": "lora-sentence-bare-r2",
        "note": "LoRA · 규칙서 없이 최종 문장만 학습",
        "detail": "형식 점수(AM) 평균이 99.9%로 표에서 제일 높은데 135건 전부 "
                  "negative다. **점수가 높다고 배운 것이 아니다**를 한 화면에서 "
                  "보여주는 자리다",
    },
    {
        "key": "kormo-bad-positive", "engine": "kormo-lora",
        "adapter": "lora-full-nonegative-r2",
        "label": "이상함 ② 아니라고 못 한다",
        "run": "lora-full-nonegative-r2",
        "note": "LoRA · 학습에서 negative를 통째로 뺐다",
        "detail": "135건 전부 positive다. 안 바뀐 조문을 줘도 바뀌었다고 답한다. "
                  "학습 데이터에 없던 답은 낼 줄 모른다",
    },
    {
        "key": "kormo-bad-broken", "engine": "kormo-lora",
        "adapter": "lora-full-down120-s43",
        "label": "이상함 ③ 말이 깨진다",
        "run": "lora-full-down120-s43",
        "note": "LoRA · 학습 562건 · 계열당 120건으로 깎음 · seed 43",
        "detail": "135건 중 70건은 판정조차 안 읽힌다. 형식이 무너진 실패라 "
                  "앞의 둘과 고장 난 자리가 다르다",
    },
]

# 어느 원본 위에 올릴지. `dtype`은 bf16 고정이다 -- 학습과 채점이 전부 bf16이었으므로
# 여기서만 다른 정밀도를 쓰면 답이 미묘하게 달라진다.
ENGINES = {
    "kormo-delora": {"model": KORMO, "label": "KORMo + DeLoRA"},
    "kormo-lora": {"model": KORMO, "label": "KORMo + LoRA 3개"},
    "qwen": {"model": QWEN, "label": "Qwen + DeLoRA"},
}


class Engine:
    """원본 하나와 거기 붙은 어댑터들. **생성은 한 번에 하나씩만 돌린다.**

    같은 GPU에 요청 둘이 겹치면 메모리가 두 배로 들고, 무엇보다 `set_adapter`로 이름을
    갈아 끼우는 구조라 **둘이 겹치면 서로의 어댑터로 답한다.** 자물쇠가 그것을 막는다.
    """

    def __init__(self, name: str, model_id: str, device: str):
        self.name = name
        self.model_id = model_id
        self.device = device
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.adapters: list[str] = []
        self.error: str | None = None

    def load(self, adapters: list[tuple[str, Path]]) -> None:
        import torch

        print(f"\n[{self.name}] {self.model_id} -> {self.device}")
        self.tokenizer = load_tokenizer(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = load_causal_lm(self.model_id, dtype=torch.bfloat16,
                               device_map=self.device)

        if adapters:
            from peft import PeftModel
            first_name, first_path = adapters[0]
            print(f"  어댑터 {first_name}")
            model = PeftModel.from_pretrained(model, str(first_path),
                                              adapter_name=first_name)
            for adapter_name, path in adapters[1:]:
                print(f"  어댑터 {adapter_name}")
                model.load_adapter(str(path), adapter_name=adapter_name)
            self.adapters = [name for name, _ in adapters]

        model.eval()
        self.model = model
        print(f"[{self.name}] 준비됨")

    def generate(self, prompt: str, adapter: str | None, max_new_tokens: int) -> dict:
        import torch

        with self.lock:
            if adapter is None:
                # 어댑터를 끄면 남는 것이 정확히 학습 전 원본이다. 원본을 따로 싣지
                # 않는 이유이기도 하다.
                context = self.model.disable_adapter()
            else:
                self.model.set_adapter(adapter)
                context = nullcontext()

            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            started = time.time()
            with context, torch.no_grad():
                generated = self.model.generate(
                    **encoded, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=None,
                    pad_token_id=self.tokenizer.pad_token_id)
            elapsed = time.time() - started

        # 프롬프트를 되풀이한 부분은 잘라낸다. 안 자르면 프롬프트 안의 예시 JSON이
        # 모델 답으로 읽힌다. `cli/evaluate.py`와 같은 처리다.
        fresh = generated[0][encoded["input_ids"].shape[1]:]
        return {
            "text": self.tokenizer.decode(fresh, skip_special_tokens=True),
            "new_tokens": int(fresh.shape[0]),
            "prompt_tokens": int(encoded["input_ids"].shape[1]),
            "seconds": round(elapsed, 2),
        }


def adapter_path(run: str) -> Path:
    return ROOT / "runs" / run / "final"


def build_engines(keys: list[str], kormo_device: str, qwen_device: str) -> dict:
    """고른 칸에 필요한 원본만 싣는다. **하나가 안 떠도 나머지는 쓴다.**

    어댑터가 없거나 GPU가 모자라 한 벌이 못 뜨는 일은 흔한데, 그때 서버가 통째로
    죽으면 나머지 다섯 칸까지 못 쓰게 된다. 못 뜬 이유를 칸에 담아 페이지가 그 칸만
    잠그게 한다.
    """
    wanted: dict[str, list[tuple[str, Path]]] = {}
    for entry in CATALOG:
        if entry["key"] not in keys:
            continue
        pairs = wanted.setdefault(entry["engine"], [])
        if entry["adapter"] and entry["adapter"] not in [n for n, _ in pairs]:
            pairs.append((entry["adapter"], adapter_path(entry["adapter"])))

    engines: dict[str, Engine] = {}
    for name, adapters in wanted.items():
        device = qwen_device if name == "qwen" else kormo_device
        engine = Engine(name, ENGINES[name]["model"], device)
        missing = [str(path) for _, path in adapters if not path.exists()]
        if missing:
            engine.error = "어댑터 폴더가 없습니다: " + ", ".join(missing)
            print(f"\n[{name}] 건너뜀 -- {engine.error}")
        else:
            try:
                engine.load(adapters)
            except Exception as error:  # noqa: BLE001 -- 한 벌이 죽어도 나머지는 쓴다
                engine.error = f"{type(error).__name__}: {error}"
                print(f"\n[{name}] 못 실었습니다 -- {engine.error}")
                traceback.print_exc()
        engines[name] = engine
    return engines


def catalog_status(engines: dict, keys: list[str]) -> list[dict]:
    rows = []
    for entry in CATALOG:
        if entry["key"] not in keys:
            continue
        engine = engines.get(entry["engine"])
        ready = bool(engine and engine.model is not None)
        rows.append({
            **{k: entry[k] for k in ("key", "label", "run", "note", "detail")},
            "model": ENGINES[entry["engine"]]["model"],
            "engine": entry["engine"],
            "ready": ready,
            "error": None if ready else (engine.error if engine else "안 실었습니다"),
        })
    return rows


def make_handler(engines: dict, keys: list[str], default_max_new_tokens: int):
    entries = {entry["key"]: entry for entry in CATALOG}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 요청마다 한 줄. 기본 형식은 너무 길다
            print(f"  {self.address_string()} {fmt % args}")

        def _cors(self) -> None:
            # 페이지는 file:// 이고 서버는 localhost:8000 이라 **교차 출처**다.
            # 이 머리글이 없으면 브라우저가 답을 받아 놓고 스크립트에 안 넘긴다.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802 -- http.server가 정한 이름이다
            # 브라우저가 본 요청 전에 "보내도 되나" 물어보는 예비 요청(preflight)이다.
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("", "/models", "/health"):
                self._send(200, {"ok": True, "models": catalog_status(engines, keys)})
            else:
                self._send(404, {"ok": False, "error": f"없는 주소입니다: {self.path}"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._send(404, {"ok": False, "error": f"없는 주소입니다: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception as error:  # noqa: BLE001
                self._send(400, {"ok": False, "error": f"본문을 못 읽었습니다: {error}"})
                return

            key = body.get("model")
            entry = entries.get(key)
            if entry is None or key not in keys:
                self._send(400, {"ok": False, "error": f"없는 모델입니다: {key}"})
                return
            engine = engines.get(entry["engine"])
            if engine is None or engine.model is None:
                self._send(503, {"ok": False,
                                 "error": engine.error if engine else "안 실었습니다"})
                return

            before, after = body.get("before") or "", body.get("after") or ""
            if not before.strip() or not after.strip():
                self._send(400, {"ok": False, "error": "개정 전과 개정 후를 둘 다 주세요"})
                return

            record = {
                "before_id": body.get("before_id") or "개정 전",
                "after_id": body.get("after_id") or "개정 후",
                "before": before, "after": after,
            }
            prefill = body.get("prefill", PREFILL)
            max_new_tokens = int(body.get("max_new_tokens") or default_max_new_tokens)
            try:
                prompt = build_prompt(record, rules=body.get("rules", True))
                result = engine.generate(prompt + prefill, entry["adapter"],
                                         max_new_tokens)
            except Exception as error:  # noqa: BLE001 -- 한 건이 죽어도 서버는 산다
                traceback.print_exc()
                self._send(500, {"ok": False,
                                 "error": f"{type(error).__name__}: {error}"})
                return

            # 프리필은 우리가 넣어 준 것이지만 답의 일부다. 다시 이어 붙여야 여는
            # 중괄호를 갖춘 온전한 JSON이 되어 파서가 제대로 읽는다.
            raw = prefill + result["text"]
            self._send(200, {
                "ok": True, "model": key, "raw": raw,
                "parsed": parse_output(raw),
                "new_tokens": result["new_tokens"],
                "prompt_tokens": result["prompt_tokens"],
                "seconds": result["seconds"],
                "truncated": result["new_tokens"] >= max_new_tokens,
            })

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="기본은 이 기계 안에서만 열린다. 밖에서 바로 붙이지 말고 "
                         "ssh -L 로 굴을 파는 것이 안전하다")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"실을 칸을 고른다. 기본은 전부. 고를 수 있는 것: "
                         f"{' '.join(e['key'] for e in CATALOG)}")
    ap.add_argument("--kormo-device", default="cuda:0")
    ap.add_argument("--qwen-device", default="cuda:1",
                    help="27B라 한 장을 거의 다 쓴다. KORMo와 같은 장에 두지 않는다")
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="cli/evaluate.py의 기본값과 같다")
    ap.add_argument("--list", action="store_true", help="안 싣고 목록만 찍는다")
    args = ap.parse_args()

    keys = [entry["key"] for entry in CATALOG]
    if args.only:
        unknown = [k for k in args.only if k not in keys]
        if unknown:
            raise SystemExit(f"모르는 칸입니다: {' '.join(unknown)}\n"
                             f"고를 수 있는 것: {' '.join(keys)}")
        keys = [k for k in keys if k in args.only]

    if args.list:
        for entry in CATALOG:
            mark = "o" if entry["key"] in keys else "-"
            path = adapter_path(entry["adapter"]) if entry["adapter"] else None
            state = "원본" if path is None else ("있음" if path.exists() else "없음")
            print(f" {mark} {entry['key']:<20} {entry['label']:<22} "
                  f"어댑터 {state:<4} {entry['adapter'] or ''}")
        return

    # 이 서버는 8장을 여럿이 나눠 쓴다. 안 주면 0번을 잡는데 0번은 우리 몫이 아니다.
    # `cli/train.py`·`cli/evaluate.py`와 같은 자리에서 같은 이유로 막는다.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES를 지정하세요. 이 프로젝트 몫은 4·5번입니다.\n"
            "  CUDA_VISIBLE_DEVICES=4,5 python -m cli.serve --port 8000")

    engines = build_engines(keys, args.kormo_device, args.qwen_device)

    print("\n" + "=" * 62)
    for row in catalog_status(engines, keys):
        mark = "o" if row["ready"] else "x"
        tail = "" if row["ready"] else f"  <- {row['error']}"
        print(f" {mark} {row['key']:<20} {row['label']}{tail}")
    ready = sum(1 for row in catalog_status(engines, keys) if row["ready"])
    print("=" * 62)
    print(f"\n{ready}/{len(keys)}칸 준비됨.  http://{args.host}:{args.port}/models")
    if ready == 0:
        print("\n! 한 칸도 못 실었습니다. 위의 이유를 보세요.")
    print("\n로컬에서 브라우저가 닿게 하려면 (켜 둔 채로 페이지를 엽니다):")
    print(f"  ssh -L {args.port}:localhost:{args.port} <서버주소>")
    print("\n멈추려면 Ctrl+C")

    handler = make_handler(engines, keys, args.max_new_tokens)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n내렸습니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
