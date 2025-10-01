
from typing import List

def chunk_text(s: str, max_tokens=180, overlap=30) -> List[str]:
    """
    Split a string into overlapping word chunks.

    Args:
        s: The input text (already cleaned).
        max_tokens: Maximum words per chunk.
        overlap: Number of words overlapping between consecutive chunks.

    Returns:
        List[str]: A list of chunk strings.
    """
    words = s.split()
    if not words: return []
    step = max_tokens - overlap
    return [" ".join(words[i:i+max_tokens]) for i in range(0, max(1, len(words)-overlap), step)]
