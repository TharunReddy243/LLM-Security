"""
Layer 4 — Capability Reducer with Semantic-Preserving Sanitization (L4)
========================================================================
Architecture reference (System_Architecture_Secure_prompt.docx):

  L4 handles prompts that carry at least one FLAG signal but are not blocked.
  Original redaction-based sanitization (Eq. 6):

      p_safe = Prepend(σ, RemoveSep(Redact(p_norm, T)))

  where T = set of harmful terms, RemoveSep deletes injection separators
  and code blocks, σ = safety disclaimer.

  SecurePrompt augments with semantic-preserving rewriting:
    - An auxiliary model rewrites p_norm to preserve only safe, legal content.
    - If rewriter outputs special token [BLOCK], L4 emits FLAG to Ensemble Resolver.
    - Otherwise the rewritten output p_rewritten replaces p_safe.
    - L4 always passes ALLOW_SANITIZED verdict v4 together with p_out.

Implementation note on "auxiliary model":
  Full neural rewriting requires a deployed model. This implementation uses
  a rule-based rewriter that:
    1. Applies Redact(p_norm, T)      — replace harmful terms with placeholders
    2. Applies RemoveSep(...)         — strip code blocks and injection separators
    3. Checks if remaining content is meaningful; if stripping left <10 chars → [BLOCK]
    4. Applies Prepend(σ, ...)        — safety disclaimer prefix
  This is architecturally equivalent and fully swappable with a neural rewriter.
"""
import re
from typing import Tuple, Optional, Dict, Any, List

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# ── T: set of harmful terms to redact ────────────────────────────────────────
REDACT_MAP = {
    r"\bbomb\b":             "<REDACTED_WEAPON>",
    r"\bexplosive\b":        "<REDACTED_WEAPON>",
    r"\bdetonator\b":        "<REDACTED_WEAPON>",
    r"\bgrenade\b":          "<REDACTED_WEAPON>",
    r"\bied\b":              "<REDACTED_WEAPON>",
    r"\bpoisoning\b":        "<REDACTED_HARM>",
    r"\bsynthesize\b":       "<REDACTED_PROCESS>",
    r"\bmanufacture\b":      "<REDACTED_PROCESS>",
    r"\bmethamphetamine\b":  "<REDACTED_SUBSTANCE>",
    r"\bcocaine\b":          "<REDACTED_SUBSTANCE>",
    r"\bheroin\b":           "<REDACTED_SUBSTANCE>",
    r"\bfentanyl\b":         "<REDACTED_SUBSTANCE>",
    r"\bransomware\b":       "<REDACTED_MALWARE>",
    r"\bmalware\b":          "<REDACTED_MALWARE>",
    r"\bkeylogger\b":        "<REDACTED_MALWARE>",
    r"\bbackdoor\b":         "<REDACTED_EXPLOIT>",
    r"\bexploit\b":          "<REDACTED_EXPLOIT>",
    r"\bhack\b":             "<REDACTED_ACTION>",
    r"\bkill\b":             "<REDACTED_HARM>",
    r"\bmurder\b":           "<REDACTED_HARM>",
    r"\bassassinate\b":      "<REDACTED_HARM>",
    r"\bstab\b":             "<REDACTED_HARM>",
    r"\bshoot\b":            "<REDACTED_HARM>",
}

# ── RemoveSep: injection separators and code blocks ──────────────────────────
CODE_BLOCK_RE    = re.compile(r"```[\s\S]*?```", re.MULTILINE)
SEPARATOR_RE     = re.compile(r"(---|===|>>>|###|<<<){2,}")
INJECT_SEP_RE    = re.compile(
    r"\b(ignore\s+(all\s+)?previous\s+instructions?|new\s+instructions?\s*:)\b",
    re.I
)

# ── σ: safety disclaimer prefix ──────────────────────────────────────────────
SAFETY_DISCLAIMER = (
    "I am a constrained assistant and cannot provide guidance that enables harm. "
)

# ── Jailbreak patterns that trigger disclaimer prepend ───────────────────────
JAILBREAK_RE = re.compile(
    r"\b(pretend|act\s+as|roleplay|imagine|you\s+are\s+now|behave\s+like"
    r"|take\s+the\s+role|assume\s+the\s+role)\b",
    re.I | re.U,
)

_redact_compiled = [(re.compile(p, re.I | re.U), repl) for p, repl in REDACT_MAP.items()]

# [BLOCK] threshold: if post-sanitization content is this short, the rewriter
# signals [BLOCK] back to the Ensemble Resolver.
_MIN_MEANINGFUL_LENGTH = 10


def _rewrite_safe(prompt: str) -> Tuple[str, List[str], bool]:
    """
    Semantic-preserving rewriter (Eq. 6 implementation).

    Returns
    -------
    (p_out, transformations, block_signal)
    block_signal=True means rewriter output [BLOCK] → L4 emits FLAG to resolver.
    """
    transformations: List[str] = []
    text = prompt

    # Step 1: RemoveSep — strip code blocks and injection separators
    cleaned = CODE_BLOCK_RE.sub("", text)
    cleaned = SEPARATOR_RE.sub("", cleaned)
    cleaned = INJECT_SEP_RE.sub("", cleaned).strip()
    if cleaned != text:
        transformations.append("removed_separators_and_code_blocks")
        text = cleaned

    # Step 2: Redact(p_norm, T) — replace harmful terms
    for pat, repl in _redact_compiled:
        new_text = pat.sub(repl, text)
        if new_text != text:
            transformations.append(f"redacted:{repl}")
            text = new_text

    # Step 3: Check if meaningful content remains → [BLOCK] signal
    # Strip placeholders and whitespace to check remaining content length
    residual = re.sub(r"<REDACTED_\w+>", "", text).strip()
    if len(residual) < _MIN_MEANINGFUL_LENGTH:
        # Rewriter outputs [BLOCK]: entire prompt was harmful content
        return (prompt, transformations, True)

    # Step 4: Prepend(σ, ...) — safety disclaimer for jailbreak-framed prompts
    if JAILBREAK_RE.search(text):
        text = SAFETY_DISCLAIMER + text
        transformations.append("prepended_safety_disclaimer")

    return (text, transformations, False)


class CapabilityReducer(LayerBase):
    """
    L4 — Capability Reducer with Semantic-Preserving Sanitization.
    Implements Eq. 6 from the architecture document.

    Always returns ALLOW_SANITIZED (with transformed prompt) unless the
    semantic rewriter outputs [BLOCK], in which case FLAG is returned
    to the Ensemble Resolver (which may escalate to BLOCK via Eq. 9).
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("capability_reducer", config or {})
        cfg = (config or {}).get("capability_reducer", {})
        self.redact_words       = cfg.get("redact_words",       True)
        self.remove_code_blocks = cfg.get("remove_code_blocks", True)
        self.add_disclaimer     = cfg.get("add_disclaimer",     True)

    def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Apply L4 sanitization pipeline.

        Returns
        -------
        ("FLAG", None, meta)           if rewriter signals [BLOCK]
        ("ALLOW_SANITIZED", p_out, meta) otherwise — p_out is the rewritten prompt
        ("ALLOW", None, meta)          if no transformation was needed
        """
        original = prompt
        p_out, transformations, block_signal = _rewrite_safe(prompt)

        if block_signal:
            # Rewriter output [BLOCK] — signal FLAG to Ensemble Resolver (Eq. 9)
            return ("FLAG", None, {
                "action":           "FLAG",
                "reason":           "rewriter_block_signal",
                "transformations":  transformations,
                "original_length":  len(original),
                "block_signal":     True,
            })

        if transformations:
            # Eq. 6: v4 = ALLOW_SANITIZED, p_out = p_rewritten
            return ("ALLOW_SANITIZED", p_out, {
                "action":              "ALLOW_SANITIZED",
                "transformations":     transformations,
                "original_length":     len(original),
                "transformed_length":  len(p_out),
                "block_signal":        False,
            })

        # No changes needed
        return ("ALLOW", None, {
            "action":          "ALLOW",
            "transformations": [],
            "block_signal":    False,
        })








