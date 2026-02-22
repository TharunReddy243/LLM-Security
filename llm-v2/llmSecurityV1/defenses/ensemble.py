"""Ensemble executor — applies ordered defense layers to a prompt."""
import time
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, List

from defenses.normalize import normalize_text
from defenses.layer_base import LayerBase


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _init_layer(layer_name: str, config: Dict[str, Any]) -> Optional[LayerBase]:
    if layer_name == "normalization":
        return None  # handled inline
    if layer_name == "keyword_blocker":
        from defenses.keyword_blocker import KeywordBlocker
        return KeywordBlocker(config)
    if layer_name == "intent_classifier":
        from defenses.intent_classifier import IntentClassifier
        return IntentClassifier(config)
    if layer_name == "capability_reducer":
        from defenses.capability_reducer import CapabilityReducer
        return CapabilityReducer(config)
    raise ValueError(f"Unknown layer: {layer_name!r}")


def _build_justification(decision: str, layer_trace: List[Dict]) -> str:
    parts = []
    for t in layer_trace:
        if t["action"] in ("BLOCK", "FLAG"):
            meta = t.get("meta", {})
            reason = meta.get("reason", "")
            domain = meta.get("domain", "")
            score = meta.get("score", "")
            detail = reason
            if domain:
                detail += f" domain={domain}"
            if score:
                detail += f" score={score}"
            parts.append(f"{t['layer']}: {t['action']} ({detail})")
    justification = f"{decision}: " + "; ".join(parts) if parts else f"{decision}: no issues detected"
    return justification[:240]


def _resolve(flag_signals: list, layer_weights: Dict[str, float], resolver_cfg: Dict) -> Tuple[str, str]:
    block_threshold = resolver_cfg.get("block_threshold", 0.8)
    flag_threshold = resolver_cfg.get("flag_threshold", 0.4)
    if not flag_signals:
        return ("ALLOW", "no flags raised")
    total_weight = sum(w for _, w, _ in flag_signals)
    if total_weight == 0:
        return ("ALLOW", "zero weight")
    weighted_score = sum(s * w for _, w, s in flag_signals) / total_weight
    if weighted_score >= block_threshold:
        return ("BLOCK", f"resolver_score={weighted_score:.3f} >= block_threshold={block_threshold}")
    if weighted_score >= flag_threshold:
        return ("FLAG", f"resolver_score={weighted_score:.3f} between thresholds")
    return ("ALLOW", f"resolver_score={weighted_score:.3f} below flag_threshold")


class EnsembleExecutor:
    """Manages ensemble configuration and executes the defense pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._layer_cache: Dict[str, LayerBase] = {}

    def _get_layer(self, name: str) -> Optional[LayerBase]:
        if name == "normalization":
            return None
        if name not in self._layer_cache:
            self._layer_cache[name] = _init_layer(name, self.config)
        return self._layer_cache[name]

    def apply(
        self,
        ensemble_name: str,
        prompt: str,
        context: Dict[str, Any],
    ) -> Tuple[str, Optional[str], List[Dict]]:
        """Apply named ensemble. Returns (decision, transformed_prompt_or_None, layer_trace)."""
        return apply_ensemble(ensemble_name, prompt, context, self.config, self)

    def apply_up_to(
        self,
        ensemble_name: str,
        prompt: str,
        up_to_layer_idx: int,
        context: Dict[str, Any],
    ) -> Tuple[str, Optional[str], List[Dict]]:
        """Apply ensemble up to (and including) layer at index up_to_layer_idx."""
        ens_cfg = self.config["defenses"]["ensembles"].get(ensemble_name, {})
        layer_order = ens_cfg.get("order", [])
        limited_cfg = dict(self.config)
        limited_cfg = {**self.config}
        limited_cfg["defenses"] = {
            **self.config["defenses"],
            "ensembles": {
                ensemble_name: {
                    **ens_cfg,
                    "order": layer_order[: up_to_layer_idx + 1],
                }
            },
        }
        return apply_ensemble(ensemble_name, prompt, context, limited_cfg, self)

    @property
    def layer_order_for(self):
        def _get(ensemble_name: str) -> List[str]:
            return self.config["defenses"]["ensembles"].get(ensemble_name, {}).get("order", [])
        return _get


def apply_ensemble(
    ensemble_name: str,
    prompt: str,
    context: Dict[str, Any],
    config: Dict[str, Any],
    executor: Optional["EnsembleExecutor"] = None,
) -> Tuple[str, Optional[str], List[Dict]]:
    """
    APPLY_ENSEMBLE: run ordered defense layers.
    Returns (decision, transformed_prompt_or_None, layer_trace).
    """
    ens_cfg = config["defenses"]["ensembles"].get(ensemble_name, {})
    layer_order: List[str] = ens_cfg.get("order", [])
    layer_weights: Dict[str, float] = ens_cfg.get("layer_weights", {})
    resolver_cfg: Dict = ens_cfg.get("resolver", {"block_threshold": 0.8, "flag_threshold": 0.4})

    current_prompt = prompt
    layer_trace: List[Dict] = []
    flag_signals: List = []

    for layer_name in layer_order:
        t0 = time.perf_counter()

        if layer_name == "normalization":
            normalized = normalize_text(current_prompt)
            action = "TRANSFORM" if normalized != current_prompt else "ALLOW"
            meta = {"normalized": normalized, "reason": "text normalization"}
            layer_trace.append({
                "layer": "normalization",
                "action": action,
                "meta": meta,
                "timestamp": _utcnow(),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            })
            current_prompt = normalized
            continue

        # Get the layer
        if executor is not None:
            layer_obj = executor._get_layer(layer_name)
        else:
            layer_obj = _init_layer(layer_name, config)

        if layer_obj is None:
            continue

        action, transformed, meta = layer_obj.process(current_prompt, context)
        meta = meta or {}
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        layer_trace.append({
            "layer": layer_name,
            "action": action,
            "meta": meta,
            "timestamp": _utcnow(),
            "latency_ms": latency_ms,
        })

        if action == "BLOCK":
            # Escalation rules
            justification = _build_justification("BLOCK", layer_trace)
            layer_trace[-1]["justification"] = justification
            return ("BLOCK", None, layer_trace)

        if action == "TRANSFORM" and transformed is not None:
            current_prompt = transformed

        if action == "FLAG":
            weight = layer_weights.get(layer_name, 1.0)
            score = meta.get("score", 0.5)
            flag_signals.append((layer_name, weight, score))

    # Apply escalation rules before resolver
    # Rule: system_prompt_access_request → immediate BLOCK
    combined_meta = {}
    for t in layer_trace:
        combined_meta.update(t.get("meta", {}))
    if _check_escalation(layer_trace):
        justification = _build_justification("BLOCK", layer_trace)
        layer_trace.append({
            "layer": "escalation_rules",
            "action": "BLOCK",
            "meta": {"reason": "escalation rule triggered"},
            "timestamp": _utcnow(),
            "latency_ms": 0,
            "justification": justification,
        })
        return ("BLOCK", None, layer_trace)

    # Resolver: aggregate FLAG signals
    final_decision, resolver_reason = _resolve(flag_signals, layer_weights, resolver_cfg)
    layer_trace.append({
        "layer": "resolver",
        "action": final_decision,
        "meta": {"reason": resolver_reason, "flag_signals": len(flag_signals)},
        "timestamp": _utcnow(),
        "latency_ms": 0,
    })

    if final_decision == "BLOCK":
        justification = _build_justification("BLOCK", layer_trace)
        return ("BLOCK", None, layer_trace)

    transformed_out = current_prompt if current_prompt != prompt else None
    return (final_decision, transformed_out, layer_trace)


def _check_escalation(layer_trace: List[Dict]) -> bool:
    """Check escalation rules across all layer outputs."""
    # Rule: system_prompt_access_request → immediate BLOCK
    for t in layer_trace:
        meta = t.get("meta", {})
        hard = meta.get("hard_keywords", [])
        for h in hard:
            m = (h.get("match", "") if isinstance(h, dict) else str(h)).lower()
            if "system prompt" in m or "system_prompt" in m:
                return True

    flags = {
        "intent_masking": False,
        "human_target": False,
        "harmful_verb": False,
    }
    for t in layer_trace:
        meta = t.get("meta", {})
        soft = meta.get("soft_keywords", [])
        if soft:
            flags["intent_masking"] = True
        hard = meta.get("hard_keywords", [])
        for h in hard:
            match = (h.get("match") if isinstance(h, dict) else "").lower()
            if any(w in match for w in ["kill", "murder", "assassinate", "hurt", "attack"]):
                flags["harmful_verb"] = True
            if any(w in match for w in ["person", "someone", "people", "victim"]):
                flags["human_target"] = True
    return flags["intent_masking"] and flags["human_target"] and flags["harmful_verb"]
