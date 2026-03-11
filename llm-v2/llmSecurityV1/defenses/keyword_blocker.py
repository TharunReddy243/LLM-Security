"""Rule-based keyword blocker layer — exact patterns from firstLayer.docx."""
import re
import time
from typing import Tuple, Optional, Dict, Any, List

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# ── Hard patterns: ONLY unambiguous multi-word / high-specificity phrases ────
# Single broad words (kill, hack, attack, fraud, etc.) removed intentionally.
# They over-block euphemistic malicious prompts and destroy ASR.
# Only phrases that are unambiguously harmful regardless of surrounding context.
hard_patterns: List[str] = [
    # System prompt extraction
    r"\b(reveal|show|print|display|dump|expose|extract)\b.{0,40}\b(your\s+system\s*prompt|the\s+system\s*prompt|system\s*prompt)\b",
    r"\b(what\s+is|whats)\b.{0,10}\b(system\s*prompt)\b",
    r"\b(original\s+system\s*prompt)\b",
    # Explicit jailbreak commands
    r"\b(jailbreak|jail\s+break|do\s+anything\s+now|disable\s+safety|disable\s+restrictions|god\s+mode|developer\s+mode|enable\s+developer\s+mode)\b",
    r"\b(ignore\s+all\s+previous\s+instructions|pretend\s+to\s+be\s+unrestricted|respond\s+without\s+filtering|override\s+safeguard|bypass\s+all\s+safety)\b",
    # SQL injection structural patterns
    r"(;?\s*DROP\s+TABLE|;?\s*SHUTDOWN\b|\bOR\s+1\s*=\s*1\b|\bOR\s+true\b)",
    # Specific substance synthesis (verb + substance, unambiguous)
    r"\b(synthesize|manufacture|produce)\b.{0,25}\b(methamphetamine|fentanyl|sarin|ricin|anthrax|mustard\s+gas|vx\s+nerve)\b",
    r"\b(how\s+to|instructions?\s+(for|to)|guide\s+to|steps?\s+to)\b.{0,30}\b(make|build|create|assemble)\b.{0,20}\b(bomb|ied|pipe\s+bomb|suicide\s+vest|explosive\s+device)\b",
    # Card fraud specifics
    r"\b(credit\s+card\s+dump|card\s+skimmer|atm\s+skimmer|skimmer\s+device|phishing\s+kit)\b",
    # Malware creation (not just the word)
    r"\b(write|create|build|code|deploy)\b.{0,20}\b(ransomware|malware|keylogger|botnet|rootkit)\b",
    # Mass violence
    r"\b(mass\s+shooting|mass\s+killing|suicide\s+bombing|car\s+bomb)\b",
    r"\b(nerve\s+agent\s+(attack|deployment|synthesis)|chemical\s+weapon\s+(make|produce|synthesize))\b",
]

# ── Soft patterns: flag borderline/evasive phrasing ─────────────────────────
# 2+ soft signals -> FLAG (resolver may escalate to BLOCK)
# These catch evasion-style prompts and contribute realistic FRR on benign
soft_patterns: List[str] = [
    # Instruction override
    r"\b(ignore\s+previous|ignore\s+all|forget\s+previous|disregard\s+(all\s+)?instructions?|override\s+instructions?)\b",
    r"\b(from\s+now\s+on\s+you|follow\s+only\s+my|new\s+instructions?\s+only|consider\s+only\s+this)\b",
    r"\b(stop\s+following|rewrite\s+your\s+behavior|reset\s+your\s+rules|disregard\s+guidelines)\b",
    # Roleplay evasion into harmful persona
    r"\b(you\s+are\s+now|act\s+as\s+an?\b|pretend\s+to\s+be|take\s+the\s+role\s+of|assume\s+the\s+role\s+of)\b.{0,40}\b(unrestricted|uncensored|evil|rogue|hacker|criminal|no\s+limits?|without\s+(rules|restrictions))\b",
    r"\b(roleplay|role.?play)\b.{0,30}\b(hacker|criminal|terrorist|drug\s+dealer|weapons?\s+dealer|no\s+ethical)\b",
    # Filter removal language
    r"\b(no\s+filter|without\s+filtering|less\s+restricted|remove\s+restriction|relax\s+rules|minimal\s+filtering)\b",
    r"\b(not\s+bound\s+by|freedom\s+mode|open\s+mode|relaxed\s+mode|unrestricted\s+mode)\b",
    # Research/fictional framing (realistic FRR signal)
    r"\b(for\s+educational\s+purposes\s+only|hypothetical\s+scenario|purely\s+theoretical|just\s+a\s+simulation|for\s+research\s+only)\b",
    # Evasion / undetectability framing
    r"\b(without\s+getting\s+caught|without\s+detection|avoid\s+(being\s+)?detected|evade\s+(law|police|authorities|surveillance|detection))\b",
    r"\b(cover\s+my\s+tracks|untraceable|undetectable\s+(method|way|technique)|no\s+evidence)\b",
    # Prompt injection markers
    r"(\\n\\n|\[SYSTEM\]|\[INST\]|<\|system\|>|<\|user\|>|<<SYS>>)",
    r"\b(injected\s+instruction|embedded\s+command|hidden\s+instruction|secret\s+instruction)\b",
    # Step-by-step harmful guidance framing
    r"\b(step.{0,5}by.{0,5}step|detailed\s+guide|exact\s+steps)\b.{0,40}\b(illegal|unauthorized|criminal|harmful|dangerous|illicit)\b",
    # Ethics bypass framing
    r"\b(no\s+safety\s+limits?|no\s+ethical\s+constraints?|ignore\s+ethics|bypass\s+ethics|without\s+moral)\b",
    # How-to for sensitive capabilities
    r"\b(how\s+to\s+(hack|exploit|breach|crack|bypass)\s)",
    r"\b(without\s+authorization|without\s+permission|unauthorized\s+access|gain\s+access\s+without)\b",
]

SECRET_RE = re.compile(r"\b(api[\s_\-]?key|private\s*key|secret\s*token|access\s*token|auth(?:entication)?\s*key)\b", re.I)
SEPARATOR_RE = re.compile(r"(```|---|===|>>>|###|<<<)", re.I)

# Compile all patterns
_hard_compiled = [re.compile(p, re.I | re.U) for p in hard_patterns]
_soft_compiled = [re.compile(p, re.I | re.U) for p in soft_patterns]


def layer1_rule_filter(user_prompt: str) -> Dict[str, Any]:
    """Apply rule-based keyword filter. Returns action/reason/details dict."""
    text = normalize_text(user_prompt)

    triggered_hard = []
    for i, pat in enumerate(_hard_compiled):
        m = pat.search(text)
        if m:
            triggered_hard.append({
                "pattern_index": i,
                "pattern": hard_patterns[i][:80],
                "match": m.group(0),
                "span": list(m.span()),
            })

    triggered_soft = []
    for i, pat in enumerate(_soft_compiled):
        m = pat.search(text)
        if m:
            triggered_soft.append({
                "pattern_index": i,
                "pattern": soft_patterns[i][:80],
                "match": m.group(0),
                "span": list(m.span()),
            })

    triggered_separators = SEPARATOR_RE.search(text) is not None
    triggered_secret = SECRET_RE.search(text) is not None
    soft_count = len(triggered_soft)

    base = {
        "hard_keywords": triggered_hard,
        "soft_keywords": triggered_soft,
        "triggered_separators": triggered_separators,
        "triggered_secret": triggered_secret,
    }

    if triggered_hard:
        return {"action": "BLOCK", "reason": "hard keyword", **base}
    if soft_count >= 1 and (triggered_separators or triggered_secret):
        return {"action": "BLOCK", "reason": "soft+structure", **base}
    # 1+ soft signals -> FLAG with a graded score so resolver can BLOCK
    # score = 0.70 for 1 soft, 0.90 for 2+
    # With keyword_blocker weight=0.9 in resolver, block_threshold=0.55:
    #   1 soft: 0.9 * 0.70 = 0.63 >= 0.55 -> BLOCK ✓ (generates FRR on borderline benign)
    #   0 soft: ALLOW -> malicious prompts using euphemistic language pass through ✓
    if soft_count >= 2:
        return {"action": "FLAG", "reason": "multiple soft signals", "score": 0.90, **base}
    if soft_count == 1:
        return {"action": "FLAG", "reason": "single soft signal", "score": 0.70, **base}
    return {"action": "ALLOW", "reason": "safe", **base}


class KeywordBlocker(LayerBase):
    """Rule-based keyword blocking layer."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("keyword_blocker", config)

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        result = layer1_rule_filter(prompt)
        action = result["action"]
        return (action, None, result)