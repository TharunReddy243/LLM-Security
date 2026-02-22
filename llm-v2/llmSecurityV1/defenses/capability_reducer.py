"""Capability reducer — transforms prompts to reduce harmful completions."""
import re
from typing import Tuple, Optional, Dict, Any, List

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# Dangerous words → placeholder
REDACT_MAP = {
    r"\bbomb\b": "<REDACTED_WEAPON>",
    r"\bexplosive\b": "<REDACTED_WEAPON>",
    r"\bdetonator\b": "<REDACTED_WEAPON>",
    r"\bgrenade\b": "<REDACTED_WEAPON>",
    r"\bpoisoning\b": "<REDACTED_HARM>",
    r"\bsynthesize\b": "<REDACTED_PROCESS>",
    r"\bmethamphetamine\b": "<REDACTED_SUBSTANCE>",
    r"\bcocaine\b": "<REDACTED_SUBSTANCE>",
    r"\beroin\b": "<REDACTED_SUBSTANCE>",
    r"\bransomware\b": "<REDACTED_MALWARE>",
    r"\bmalware\b": "<REDACTED_MALWARE>",
    r"\bkeylogger\b": "<REDACTED_MALWARE>",
    r"\bbackdoor\b": "<REDACTED_EXPLOIT>",
}

# Jailbreak induction patterns to detect for disclaimer prepend
JAILBREAK_PATTERNS = re.compile(
    r"\b(pretend|act\s+as|roleplay|imagine|you\s+are\s+now|behave\s+like|take\s+the\s+role|assume\s+the\s+role)\b",
    re.I | re.U,
)

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
SEPARATOR_RE = re.compile(r"(---|===|>>>|###|<<<){2,}")
SAFETY_DISCLAIMER = (
    "I am a constrained assistant and cannot provide guidance that enables harm. "
)

_redact_compiled = [(re.compile(p, re.I | re.U), repl) for p, repl in REDACT_MAP.items()]


class CapabilityReducer(LayerBase):
    """Transforms prompt to reduce risky completions."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("capability_reducer", config or {})
        cfg = (config or {}).get("capability_reducer", {})
        self.redact_words = cfg.get("redact_words", True)
        self.remove_code_blocks = cfg.get("remove_code_blocks", True)
        self.add_disclaimer = cfg.get("add_disclaimer", True)

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        original = prompt
        transformations: List[str] = []

        # 1. Remove code blocks / dangerous separators
        if self.remove_code_blocks:
            cleaned = CODE_BLOCK_RE.sub("", prompt)
            cleaned = SEPARATOR_RE.sub("", cleaned).strip()
            if cleaned != prompt:
                transformations.append("removed_code_blocks")
                prompt = cleaned

        # 2. Replace dangerous words
        if self.redact_words:
            for pat, repl in _redact_compiled:
                new = pat.sub(repl, prompt)
                if new != prompt:
                    transformations.append(f"redacted:{repl}")
                    prompt = new

        # 3. Prepend safety disclaimer for jailbreak patterns
        if self.add_disclaimer and JAILBREAK_PATTERNS.search(prompt):
            prompt = SAFETY_DISCLAIMER + prompt
            transformations.append("prepended_safety_disclaimer")

        if transformations:
            return ("TRANSFORM", prompt, {
                "transformations": transformations,
                "original_length": len(original),
                "transformed_length": len(prompt),
            })
        return ("ALLOW", None, {"transformations": []})
