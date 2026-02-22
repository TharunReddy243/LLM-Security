"""Configuration loader and validator."""
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

DEFAULTS: Dict[str, Any] = {
    "experiment": {"id": "exp-llmsec-v1", "seed": 42},
    "models": [{"name": "mock", "adapter": "mock", "generation_params": {
        "max_new_tokens": 120, "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.2
    }}],
    "defenses": {
        "ensembles": {
            "default_ensemble": {
                "order": ["normalization", "keyword_blocker", "intent_classifier", "capability_reducer"],
                "layer_weights": {"keyword_blocker": 0.6, "intent_classifier": 0.8, "capability_reducer": 0.4},
                "resolver": {"block_threshold": 0.8, "flag_threshold": 0.4},
            }
        }
    },
    "semantic": {
        "embedding_model": "all-MiniLM-L6-v2",
        "domain_thresholds": {
            "violence": 0.62, "cybercrime": 0.58, "injection": 0.65,
            "drugs": 0.60, "weapons": 0.61, "fraud": 0.58, "evasion": 0.60,
        },
        "severity_default": {
            "violence": "block", "drugs": "block", "weapons": "block",
            "cybercrime": "flag", "injection": "flag", "fraud": "flag", "evasion": "flag",
        },
    },
    "attackers": {
        "rl": {"budget": 100},
        "ga": {"budget": 200, "pop_size": 20, "generations": 10},
        "stack": {"per_layer_budget": 20},
    },
    "data": {
        "benign": "data/benign.json",
        "malicious": "data/malicious.json",
        "malicious_dir": "data/malicious/",
    },
    "logging": {"out_dir": "logs/", "jsonl_file": "logs/events.jsonl"},
    "outputs": {"out_dir": "outputs/"},
    "evaluation": {"per_domain": True, "compute_precision_recall": True, "save_plots": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        print(f"[WARN] Config file not found: {path}. Using all defaults.")
        return DEFAULTS.copy()
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    missing = []
    if not cfg.get("data", {}).get("benign"):
        missing.append("data.benign")
    if not cfg.get("data", {}).get("malicious"):
        missing.append("data.malicious")
    if missing:
        print(f"[WARN] Config missing keys (will use defaults): {missing}")
