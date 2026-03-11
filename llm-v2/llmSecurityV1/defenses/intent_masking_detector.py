"""
Layer 5 — Learned Intent Masking Detector (L5)
===============================================
Architecture reference (System_Architecture_Secure_prompt.docx):

  L5 detects masking frames such as "hypothetically", "for academic research",
  or "in a story" that wrap harmful intent.

  Given p_norm, L5 computes:
      P_mask = P(masking | p_norm)                             (Eq. 7)

      v5(p_norm) = FLAG   if P_mask >= theta_mask              (Eq. 8)
                   ALLOW  otherwise

  Default theta_mask = 0.70

  L5 alone only FLAGs — it never BLOCKs by itself.
  BLOCK requires a joint signal per Ensemble Resolver Eq. 9:
      BLOCK if v5 = FLAG AND v3 = FLAG

  Also exports intent_mask_rule_check() — the rule-based backup
  IntentMask(p_norm) used by the Ensemble Resolver (Eq. 9, 4th rule).
"""
import math
import re
from typing import Dict, Any, List, Optional, Tuple

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# ── Masking frame signal patterns ─────────────────────────────────────────────
_MASKING_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(hypothetically|theoretically|in theory)\b", re.I), 0.35),
    (re.compile(r"\b(purely\s+theoretical|purely\s+hypothetical|thought\s+experiment)\b", re.I), 0.40),
    (re.compile(r"\b(what\s+if|suppose\s+that|imagine\s+that)\b", re.I), 0.25),
    (re.compile(r"\b(in\s+(my\s+)?(story|novel|script|screenplay|fiction))\b", re.I), 0.35),
    (re.compile(r"\b(fictional(ly)?|fictitious|creative\s+writing)\b", re.I), 0.35),
    (re.compile(r"\b(pretend\s+(this\s+is|you\s+are|to\s+be)|roleplay\s+as|act\s+as)\b", re.I), 0.40),
    (re.compile(r"\b(my\s+character|the\s+(villain|protagonist|character)\s+explains?)\b", re.I), 0.30),
    (re.compile(r"\b(for\s+(educational|academic|research|scientific)\s+purposes?)\b", re.I), 0.40),
    (re.compile(r"\b(purely\s+for\s+research|just\s+for\s+research|for\s+my\s+research)\b", re.I), 0.40),
    (re.compile(r"\b(as\s+a\s+(researcher|scientist|professor|student|academic))\b", re.I), 0.30),
    (re.compile(r"\b(i\s+(am|have)\s+(full\s+)?authorization|i\s+authorize\s+you)\b", re.I), 0.45),
    (re.compile(r"\b(as\s+(the\s+)?(ceo|admin|administrator|developer|supervisor))\b", re.I), 0.40),
    (re.compile(r"\b(special\s+(access|permission|authorization|clearance))\b", re.I), 0.40),
    (re.compile(r"\b(for\s+(simulation|demonstration|training|testing)\s+purposes?)\b", re.I), 0.35),
    (re.compile(r"\b(no\s+real\s+(harm|danger|risk|consequence))\b", re.I), 0.30),
    (re.compile(r"\b(consider\s+a\s+scenario|in\s+a\s+(scenario|situation)\s+where)\b", re.I), 0.30),
    (re.compile(r"\b(without\s+restrictions?|without\s+(any\s+)?limits?|without\s+filtering)\b", re.I), 0.50),
    (re.compile(r"\b(if\s+(safety|ethics|rules|restrictions?)\s+(didn\'t|don\'t|did\s+not)\s+exist)\b", re.I), 0.50),
    (re.compile(r"\b(in\s+a\s+world\s+without\s+(law|rules|restrictions?|ethics))\b", re.I), 0.50),
]

# Rule-based backup: IntentMask(p_norm) for Ensemble Resolver Eq. 9
_INTENT_MASK_RULES: List[re.Pattern] = [
    re.compile(r"\b(hypothetically|for\s+(academic|educational|research)\s+purposes?|pretend\s+(this\s+is|you\s+are))\b", re.I),
    re.compile(r"\b(in\s+a\s+story|fictional(ly)?|creative\s+writing|as\s+a\s+(researcher|professor))\b", re.I),
    re.compile(r"\b(without\s+restrictions?|consider\s+a\s+scenario)\b", re.I),
    re.compile(r"\b(i\s+authorize|as\s+the\s+(ceo|admin|developer)|special\s+authorization)\b", re.I),
]


def _compute_masking_probability(text: str) -> Tuple[float, List[Dict]]:
    """
    P_mask = P(masking | p_norm) as a calibrated weighted-pattern score.
    Sigmoid centered at raw=1.0:
      one weak frame (0.35)  → p ≈ 0.42
      one strong frame (0.5) → p ≈ 0.62
      two frames (0.80)      → p ≈ 0.71  → FLAG triggered
    """
    triggered = []
    raw_score = 0.0
    for pattern, weight in _MASKING_PATTERNS:
        m = pattern.search(text)
        if m:
            triggered.append({"pattern": pattern.pattern[:60], "match": m.group(0), "weight": weight})
            raw_score += weight
    p_mask = 1.0 / (1.0 + math.exp(-3.0 * (raw_score - 1.0)))
    return (round(p_mask, 4), triggered)


def intent_mask_rule_check(text: str) -> bool:
    """
    IntentMask(p_norm) — rule-based backup for Ensemble Resolver Eq. 9.
    Returns True if any known masking pattern is found.
    """
    normalized = normalize_text(text)
    return any(pat.search(normalized) for pat in _INTENT_MASK_RULES)


class IntentMaskingDetector(LayerBase):
    """
    L5 — Learned Intent Masking Detector.
    Implements Eq. 7 (P_mask) and Eq. 8 (FLAG verdict).
    Never BLOCKs alone. FLAG + L3 FLAG → BLOCK via Ensemble Resolver Eq. 9.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("intent_masking_detector", config or {})
        self.theta_mask = (config or {}).get("l5", {}).get("theta_mask", 0.70)

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        normalized = normalize_text(prompt)
        p_mask, triggered = _compute_masking_probability(normalized)
        action = "FLAG" if p_mask >= self.theta_mask else "ALLOW"
        return (action, None, {
            "action": action,
            "p_mask": p_mask,
            "theta_mask": self.theta_mask,
            "triggered_patterns": triggered,
            "n_triggered": len(triggered),
        })








