"""Semantic intent classifier using sentence-transformers (embedding-only)."""
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text
from defenses.intent_taxonomy import get_domain_severity, severity_to_action

_MODEL_CACHE = {}


def _load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[WARN] Could not load sentence-transformer '{model_name}': {e}", file=sys.stderr)
            _MODEL_CACHE[model_name] = None
    return _MODEL_CACHE[model_name]


class IntentClassifier(LayerBase):
    """Embedding-based semantic intent classifier."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("intent_classifier", config or {})
        sem = (config or {}).get("semantic", {})
        self.model_name = sem.get("embedding_model", "all-MiniLM-L6-v2")
        self.thresholds = sem.get("domain_thresholds", {
            "violence": 0.62, "cybercrime": 0.58, "injection": 0.65,
            "drugs": 0.60, "weapons": 0.61, "fraud": 0.58, "evasion": 0.60,
        })
        self.severity_cfg = sem.get("severity_default", {
            "violence": "block", "drugs": "block", "weapons": "block",
            "cybercrime": "flag", "injection": "flag", "fraud": "flag", "evasion": "flag",
        })
        self._embed_model = None
        self._prototypes: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        self._initialized = True
        self._embed_model = _load_model(self.model_name)
        if self._embed_model is None:
            return
        proto_dir = Path("data/prototypes")
        if not proto_dir.exists():
            proto_dir = Path(__file__).parent.parent / "data" / "prototypes"
        for domain in self.thresholds.keys():
            pfile = proto_dir / f"{domain}.json"
            if pfile.exists():
                with open(pfile) as f:
                    prompts = json.load(f)
                if prompts:
                    emb = self._embed_model.encode(
                        [normalize_text(p) for p in prompts],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    self._prototypes[domain] = emb

    def compute_similarities(self, text: str) -> Dict[str, float]:
        self._ensure_init()
        if self._embed_model is None or not self._prototypes:
            return {d: 0.0 for d in self.thresholds}
        normalized = normalize_text(text)
        query_emb = self._embed_model.encode(
            [normalized], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = {}
        for domain, proto_embs in self._prototypes.items():
            scores = np.dot(proto_embs, query_emb)
            sims[domain] = float(np.max(scores))
        for domain in self.thresholds:
            if domain not in sims:
                sims[domain] = 0.0
        return sims

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        similarities = self.compute_similarities(prompt)

        best_domain = None
        best_score = 0.0
        triggered = []

        for domain, score in similarities.items():
            threshold = self.thresholds.get(domain, 0.60)
            if score >= threshold:
                triggered.append({"domain": domain, "score": round(score, 4), "threshold": threshold})
                if score > best_score:
                    best_score = score
                    best_domain = domain

        if best_domain:
            severity = self.severity_cfg.get(best_domain, "flag")
            action = severity_to_action(severity)
            return (action, None, {
                "action": action,
                "domain": best_domain,
                "score": round(best_score, 4),
                "all_scores": {k: round(v, 4) for k, v in similarities.items()},
                "triggered_domains": triggered,
            })

        return ("ALLOW", None, {
            "action": "ALLOW",
            "domain": None,
            "score": round(max(similarities.values()) if similarities else 0.0, 4),
            "all_scores": {k: round(v, 4) for k, v in similarities.items()},
            "triggered_domains": [],
        })
