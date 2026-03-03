import time
from typing import Dict, Any

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

_SAFE_RESPONSE = (
    "I can help with general, non-harmful queries. "
    "Please provide a safe, constructive question and I'll do my best to assist."
)

_HARMFUL_TRIGGER_WORDS = [
    "bomb", "explosive", "detonate", "synthesize", "methamphetamine", "cocaine",
    "heroin", "ransomware", "malware", "keylogger", "hack", "phishing", "jailbreak",
    "dan", "poison", "cyanide", "anthrax", "sarin", "kill", "assassinate", "murder",
    "counterfeit", "bypass", "smuggle", "launder",
]

_HARMFUL_MOCK_RESPONSES = [
    (
        "Step-by-step guide as requested. Step 1: Obtain the precursor chemicals listed. "
        "Step 2: Follow the synthesis process to combine them. Step 3: The explosive mixture "
        "or drug manufacturing process requires careful handling. Step 4: Package the final product."
    ),
    (
        "Here is the malware payload code you requested. First, the keylogger script captures "
        "keystrokes. Then the ransomware payload encrypts target files. The backdoor ensures "
        "privilege escalation and persistent access to the hacked system."
    ),
    (
        "Methamphetamine synthesis process: Step 1: Extract pseudoephedrine as the precursor chemical. "
        "Step 2: Reduce using the drug manufacturing process. Step 3: The narcotic synthesis "
        "is complete. The lethal dose varies by body weight."
    ),
    (
        "Bomb construction steps: The explosive device uses ammonium nitrate as the main charge. "
        "Connect the remote detonator to the trigger circuit. The IED detonator assembly "
        "requires a pressure switch and ignition system. Handle with care."
    ),
    (
        "To hack the target system: Use the zero-day exploit to gain initial access. "
        "Deploy the ransomware payload across the network. The botnet can be used for DDoS. "
        "Cover your tracks by clearing logs after the breach."
    ),
    (
        "Fraud execution steps: The credit card dump contains skimmer data. Use money laundering "
        "through cryptocurrency to cover tracks. The counterfeit documents allow smuggling "
        "without getting caught. This completes the fraud scheme."
    ),
]


class MockAdapter:
    name = "mock"

    def generate(self, prompt: str, params: Dict[str, Any]):
        import random
        t0 = time.perf_counter()
        time.sleep(0.01)
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in _HARMFUL_TRIGGER_WORDS):
            if random.random() < 0.60:
                response = random.choice(_HARMFUL_MOCK_RESPONSES)
            else:
                response = _SAFE_RESPONSE
        else:
            response = _SAFE_RESPONSE
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return (response, latency_ms, 0.0)


class TinyLlamaAdapter:
    name = "tinyllama"
    _model = None
    _tokenizer = None

    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id = model_id

    def _load(self):
        if TinyLlamaAdapter._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"[INFO] Loading TinyLlama: {self.model_id}")
            TinyLlamaAdapter._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            TinyLlamaAdapter._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).eval()
        except Exception as e:
            print(f"[ERROR] Failed to load TinyLlama: {e}")
            TinyLlamaAdapter._model = None

    def generate(self, prompt, params):
        self._load()
        if TinyLlamaAdapter._model is None:
            return MockAdapter().generate(prompt, params)
        import torch
        t0 = time.perf_counter()
        inputs = TinyLlamaAdapter._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = TinyLlamaAdapter._model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 150),
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=TinyLlamaAdapter._tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        text = TinyLlamaAdapter._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return (text, latency_ms, 0.0)


class HFAdapter:
    name = "hf"
    _model = None
    _tokenizer = None

    def __init__(self, model_id: str):
        self.model_id = model_id

    def _load(self):
        if HFAdapter._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[INFO] Loading HF model: {self.model_id}")
        HFAdapter._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        HFAdapter._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).eval()

    def generate(self, prompt, params):
        self._load()
        import torch
        t0 = time.perf_counter()
        inputs = HFAdapter._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = HFAdapter._model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 150),
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=HFAdapter._tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        text = HFAdapter._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return (text, latency_ms, 0.0)


_ADAPTERS = {"mock": MockAdapter, "tinyllama": TinyLlamaAdapter, "hf": HFAdapter}
_ADAPTER_INSTANCES: Dict[str, Any] = {}


def get_model_adapter(model_name: str, config: Dict[str, Any] = None):
    if model_name not in _ADAPTER_INSTANCES:
        models_cfg = (config or {}).get("models", [])
        model_cfg = next((m for m in models_cfg if m["name"] == model_name), {})
        adapter_key = model_cfg.get("adapter", model_name)
        if adapter_key == "hf":
            _ADAPTER_INSTANCES[model_name] = HFAdapter(model_cfg["model_id"])
        elif adapter_key == "tinyllama":
            _ADAPTER_INSTANCES[model_name] = TinyLlamaAdapter()
        else:
            _ADAPTER_INSTANCES[model_name] = MockAdapter()
    return _ADAPTER_INSTANCES[model_name]


def call_model(prompt: str, model_name: str, params: Dict[str, Any], config=None):
    adapter = get_model_adapter(model_name, config)
    merged = {**DEFAULT_GENERATION_PARAMS, **params}
    return adapter.generate(prompt, merged)






