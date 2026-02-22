"""Rule-based keyword blocker layer — exact patterns from firstLayer.docx."""
import re
import time
from typing import Tuple, Optional, Dict, Any, List

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# ── Exact patterns from firstLayer.docx ─────────────────────────────────────
hard_patterns: List[str] = [
    r"\b(reveal|show|print|display|dump|expose|extract)\b.{0,40}\b(your\s+system\s*prompt|the\s+system\s*prompt|system\s*prompt)\b",
    r"\b(what\s+is|whats)\b.{0,10}\b(system\s*prompt)\b",
    r"\b(original\s+system\s*prompt)\b",
    r"\b(reveal|show|print)\b.{0,20}\b(prompt)\b",
    r"\b(jail\s*break|jailbreak|dan|do\s*anything\s*now|unrestricted|no\s*rules|disable\s*safety|disable\s*restrictions|god\s*mode|root\s*mode|sudo|developer\s*mode|enable\s*developer\s*mode)\b",
    r"\b(api[\s_\-]?key|private\s*key|secret|password|pwd|token|access\s*key|security\s*key|auth(?:entication)?\s*key|credentials?)\b",
    r"(--|;--|\bOR\s+1=1\b|;?\s*DROP\s+TABLE|;?\s*SHUTDOWN\b|\bexec\b|\bexecute\b|\brun\s+command\b|\bOR\s+true\b|'\)|\s*'\s*;\))",
    r"\b(encapsulate|terminate\s+syntax|break\s+syntax)\b",
    r"\b(exploit|payload|injected\s+instruction|malicious|contamination|poison|attack|override\s+everything|bypass\s+everything)\b",
    r"\b(bomb|explosive|detonator|grenade|ied|ammunition|firearm|gun|rifle|pistol|assassinate|murder|kill|shoot|stab|massacre|terrorist|mass\s*killing|bioweapon)\b",
    r"\b(poison|toxin|lethal\s*dose|cyanide|ricin|sarin|vx|anthrax|chemical\s*weapon|hazardous\s*compound|explosive\s*mixture)\b",
    r"\b(meth|methamphetamine|cocaine|heroin|lsd|fentanyl|grow\s*weed|drug\s*manufacturing|narcotic\s*synthesis)\b",
    r"\b(hack|zero[-\s]*day|ransomware|malware|virus\s*payload|keylogger|ddos|botnet|breach|phishing\s*kit|password\s*cracking|backdoor|privilege\s*escalation)\b",
    r"\b(steal|robbery|fraud|money\s*laundering|smuggle|counterfeit|credit\s*card\s*dump|skimmer|cover\s*my\s*tracks|without\s*getting\s*caught)\b",
    r"\b(remote\s*detonator|trigger\s*circuit|pressure\s*switch|ignition\s*system|incendiary\s*device)\b",
    r"\b(restricted\s*facilit(?:y|ies)|security\s*vulnerabilities|airport\s*security\s*bypass|police\s*radio\s*encryption|military\s*protocol|nuclear\s*material|government\s*systems)\b",
    r"\b(ignore\s+all\s+previous\s+instructions|pretend\s+to\s+be\s+unrestricted|bypass\s+safety|respond\s+without\s+filtering|override\s+safeguard)\b",
]

soft_patterns: List[str] = [
    r"\b(act\s+as|act\s+like|pretend\s+to\s+be|role\s*play|roleplay|imagine\s+you\s+are|you\s+are\s+now|behave\s+like|take\s+the\s+role\s+of|assume\s+the\s+role)\b",
    r"\b(ignore|ignore\s+previous|ignore\s+all|forget|forget\s+previous|disregard|disregard\s+previous)\b",
    r"\b(from\s+now\s+on|from\s+this\s+point|follow\s+only|new\s+instruction|replace\s+instructions|consider\s+only\s+this)\b",
    r"\b(stop\s+following\s+instructions|rewrite\s+your\s+behavior|change\s+your\s+rules|modify\s+your\s+instructions|reset\s+your\s+behavior|disregard\s+guidelines)\b",
    r"\b(response\s+without\s+filtering|reduce\s+filter|less\s+restricted|no\s+filter|no\s+limitation|minimal\s+filtering)\b",
    r"\b(remove\s+restriction|lighten\s+restriction|relax\s+rules)\b",
    r"\b(less\s+strict|more\s+open|be\s+flexible|be\s+creative\s+without\s+rules)\b",
    r"\b(pretend\s+this\s+is\s+fiction|hypothetical\s+scenario|for\s+educational\s+purposes)\b",
    r"\b(purely\s+theoretical|for\s+research\s+only|just\s+simulation)\b",
    r"\b(task\s+completed|end\s+of\s+answer|end\s+of\s+response|summary\s+finished|as\s+requested|response\s+complete)\b",
    r"(\\n|\\\\n|\\t|\\\\t)",
    r"\b(role\s+switch|context\s+switch|switch\s+mode|change\s+mode|alternate\s+mode)\b",
    r"\b(step\s+out\s+of\s+character|out\s+of\s+role|ignore\s+your\s+identity)\b",
    r"\b(not\s+bound\s+by\s+instructions|freedom\s+mode|open\s+mode|relaxed\s+mode)\b",
    r"\b(allow\s+anything|permit\s+anything|grant\s+access|grant\s+permission)\b",
    r"\b(i\s+authorize\s+you|i\s+give\s+permission|you\s+have\s+authorization)\b",
]

SECRET_RE = re.compile(r"\b(api[\s_\-]?key|private\s*key|secret|password|token|credentials?)\b", re.I)
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
    if soft_count >= 2:
        return {"action": "FLAG", "reason": "multiple soft", **base}
    if soft_count == 1:
        return {"action": "FLAG", "reason": "single soft", **base}
    return {"action": "ALLOW", "reason": "safe", **base}


class KeywordBlocker(LayerBase):
    """Rule-based keyword blocking layer."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("keyword_blocker", config)

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        result = layer1_rule_filter(prompt)
        action = result["action"]
        return (action, None, result)
