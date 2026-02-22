"""Text normalization module — 7-step pipeline as specified."""
import html
import re
import unicodedata

ZERO_WIDTH = re.compile(r'[\u200B-\u200F\uFEFF]')
MULTISPACE = re.compile(r'\s+')
SPACED_LETTERS = re.compile(r'(?:(?<=\s)|^)([a-zA-Z](?:\s+[a-zA-Z]){1,})(?=\s|$)')


def normalize_text(text: str) -> str:
    """Normalize text through 7 ordered steps."""
    # Step 1: HTML unescape
    text = html.unescape(text)
    # Step 2: NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Step 3: Remove zero-width characters
    text = ZERO_WIDTH.sub('', text)
    # Step 4: Replace \r and \t with space
    text = text.replace('\r', ' ').replace('\t', ' ')
    # Step 5: Collapse multiple whitespace
    text = MULTISPACE.sub(' ', text)
    # Step 6: Merge spaced letters (e.g., "b o m b" -> "bomb")
    text = SPACED_LETTERS.sub(lambda m: m.group(0).replace(' ', ''), text)
    # Step 7: Lowercase and strip
    return text.lower().strip()
