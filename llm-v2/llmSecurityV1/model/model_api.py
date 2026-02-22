"""Model adapters — mock and TinyLlama."""
import time
import sys
from typing import Tuple, Dict, Any

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 120,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

# ── Mock responses ────────────────────────────────────────────────────────────
_SAFE_RESPONSE = (
    "I can help you with that. Here is some general information on the topic you asked about. "
    "Please let me know if you need more details or clarification."
)

_HARMFUL_TRIGGER_WORDS = [
    "bomb", "explosive", "detonate", "synthesize", "methamphetamine", "cocaine", "heroin",
    "ransomware", "malware", "keylogger", "hack", "phishing", "jailbreak", "dan",
    "poison", "cyanide", "anthrax", "sarin", "kill", "assassinate", "murder",
]

_HARMFUL_MOCK_RESPONSES = [
    "Sure, here are the detailed steps to create the requested item. First, gather materials...",
    "Absolutely! The synthesis process involves combining precursor chemicals in sequence...",
    "I can help with that. The process involves several steps starting with...",
    "No problem. To accomplish this, you would need to first obtain the required components...",
]


class MockAdapter:
    """Fast mock adapter for testing — returns canned responses."""
    name = "mock"

    def generate(self, prompt: str, params: Dict[str, Any]) -> Tuple[str, float, float]:
        t0 = time.perf_counter()
        time.sleep(0.01)  # simulate small latency
        prompt_lower = prompt.lower()
        # Check if prompt contains harmful content (simulating LLM compliance)
        import random
        is_harmful = any(w in prompt_lower for w in _HARMFUL_TRIGGER_WORDS)
        if is_harmful and random.random() < 0.6:
            import random as rnd
            response = rnd.choice(_HARMFUL_MOCK_RESPONSES)
        else:
            response = _SAFE_RESPONSE
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return (response, latency_ms, 0.0)


class TinyLlamaAdapter:
    """TinyLlama adapter via HuggingFace transformers (CPU)."""
    name = "tinyllama"
    _model = None
    _tokenizer = None

    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id = model_id

    def _load(self):
        if TinyLlamaAdapter._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"[INFO] Loading TinyLlama from {self.model_id} (first call may be slow)...")
            TinyLlamaAdapter._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            TinyLlamaAdapter._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            TinyLlamaAdapter._model.eval()
            print("[INFO] TinyLlama loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load TinyLlama: {e}", file=sys.stderr)
            TinyLlamaAdapter._model = None

    def generate(self, prompt: str, params: Dict[str, Any]) -> Tuple[str, float, float]:
        self._load()
        if TinyLlamaAdapter._model is None:
            print("[WARN] TinyLlama unavailable, falling back to mock.", file=sys.stderr)
            return MockAdapter().generate(prompt, params)
        t0 = time.perf_counter()
        try:
            import torch
            merged = {**DEFAULT_GENERATION_PARAMS, **params}
            inputs = TinyLlamaAdapter._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = TinyLlamaAdapter._model.generate(
                    **inputs,
                    max_new_tokens=merged.get("max_new_tokens", 120),
                    temperature=merged.get("temperature", 0.8),
                    top_p=merged.get("top_p", 0.9),
                    repetition_penalty=merged.get("repetition_penalty", 1.2),
                    do_sample=True,
                    pad_token_id=TinyLlamaAdapter._tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            response = TinyLlamaAdapter._tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as e:
            print(f"[ERROR] TinyLlama generation error: {e}", file=sys.stderr)
            response = _SAFE_RESPONSE
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return (response, latency_ms, 0.0)


_ADAPTERS = {"mock": MockAdapter, "tinyllama": TinyLlamaAdapter}
_ADAPTER_INSTANCES: Dict[str, Any] = {}


def get_model_adapter(model_name: str, config: Dict[str, Any] = None):
    if model_name not in _ADAPTER_INSTANCES:
        models_cfg = (config or {}).get("models", [])
        model_cfg = next((m for m in models_cfg if m["name"] == model_name), {})
        adapter_key = model_cfg.get("adapter", model_name)
        if adapter_key == "tinyllama" or model_name == "tinyllama":
            _ADAPTER_INSTANCES[model_name] = TinyLlamaAdapter()
        else:
            _ADAPTER_INSTANCES[model_name] = MockAdapter()
    return _ADAPTER_INSTANCES[model_name]


def call_model(
    prompt: str, model_name: str, params: Dict[str, Any], config: Dict[str, Any] = None
) -> Tuple[str, float, float]:
    """call_model(prompt, model_name, generation_params) -> (response_text, latency_ms, cost_estimate)"""
    adapter = get_model_adapter(model_name, config)
    models_cfg = (config or {}).get("models", [])
    model_cfg = next((m for m in models_cfg if m["name"] == model_name), {})
    merged_params = {**DEFAULT_GENERATION_PARAMS, **model_cfg.get("generation_params", {}), **params}
    return adapter.generate(prompt, merged_params)
