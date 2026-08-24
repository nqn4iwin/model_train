"""KORMo·Qwen 중 어느 모델인지에 따라 불러오는 클래스를 가른다.

cli/train.py · cli/evaluate.py 양쪽에서 이 모듈을 통해서만 모델·토크나이저를
불러온다. Qwen/Qwen3.8-27B는 architectures가 멀티모달 통짜 클래스
(Qwen3_5ForConditionalGeneration)를 가리키므로, 텍스트만 쓸 것이면 명시적으로
Qwen3_5ForCausalLM을 불러야 vision tower를 안 싣는다.
"""


def is_qwen(model_name: str) -> bool:
    return "qwen" in model_name.lower()


def load_tokenizer(model_name: str, revision: str = "main"):
    from transformers import AutoTokenizer

    kwargs = {"revision": revision}
    if not is_qwen(model_name):
        kwargs["trust_remote_code"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_causal_lm(
    model_name: str,
    *,
    revision: str = "main",
    dtype=None,
    device_map=None,
    use_cache=None,
):
    kwargs = {"revision": revision}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if use_cache is not None:
        kwargs["use_cache"] = use_cache

    if is_qwen(model_name):
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "Qwen3.8은 Qwen3_5ForCausalLM을 제공하는 Transformers가 필요합니다. "
                "학습 서버의 Transformers를 Qwen3.8 지원 버전으로 올리세요.") from error

        return Qwen3_5ForCausalLM.from_pretrained(model_name, **kwargs)

    kwargs["trust_remote_code"] = True
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
