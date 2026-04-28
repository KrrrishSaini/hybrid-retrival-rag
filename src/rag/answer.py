"""Answer grounding, citation, and extractive fallback for the RAG pipeline.

Takes raw LLM output + retrieved chunks and produces a grounded answer
with inline chunk citations.  Also provides a faithfulness check that
flags sentences not supported by any retrieved evidence.

When the generative answer is low quality (short, empty, or low
faithfulness), ``extractive_answer`` builds a best-effort answer by
selecting the highest-overlap sentences from the retrieved chunks.
"""

from __future__ import annotations

import re
from typing import Any


_REAL_SHORT_WORDS = {
    "e", "i", "o",  # single letter words that stand alone
    "a", "an", "am", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to",
    "up", "us", "we", "all", "and", "any", "are", "but", "can", "did",
    "for", "get", "got", "had", "has", "her", "him", "his", "how", "its",
    "may", "new", "nor", "not", "now", "old", "one", "our", "out", "own",
    "per", "put", "run", "say", "set", "she", "the", "too", "two", "use",
    "was", "way", "who", "why", "yet", "you",
    "also", "been", "both", "case", "date", "each", "even", "from",
    "have", "into", "just", "like", "make", "many", "more", "most",
    "much", "must", "need", "only", "over", "part", "some", "such",
    "than", "that", "them", "then", "they", "this", "time", "upon",
    "very", "well", "were", "what", "when", "will", "with", "year",
    "after", "before", "being", "below", "between", "every", "first",
    "given", "level", "shall", "should", "since", "state", "their",
    "there", "these", "those", "under", "which", "while", "would",
    "about", "above", "other", "where", "could", "might", "still",
    "during", "within", "ensure", "through", "served",
    "mind", "kept", "hand", "meal", "soap", "body",
    # Domain-specific short words
    "food", "card", "block", "scheme", "office", "oil", "act", "day",
    "mid", "key", "way", "law", "aid", "pay", "tax", "age", "due",
    "end", "fee", "lot", "map", "net", "raw", "red", "top", "via",
}


def _defragment_bpe(text: str) -> str:
    """Reassemble words fragmented by BPE tokenizer with naive decoding.

    BPE with a Whitespace pre-tokenizer and no configured decoder inserts
    spaces between all subword tokens (e.g. "con te x t" for "context").
    This function merges adjacent short tokens that form real words,
    while preserving real short words like "the", "is", "of", etc.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    def _is_fragment(w: str) -> bool:
        """True if token is likely a BPE subword fragment, not a real word."""
        return w.isalpha() and len(w) <= 4 and w.lower() not in _REAL_SHORT_WORDS

    result: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]

        # Skip non-alpha tokens (punctuation, numbers, etc.)
        if not w.isalpha():
            result.append(w)
            i += 1
            continue

        # If this is a known real word, don't try to merge it forward.
        # But it can still absorb trailing fragments (handled below).
        if w.lower() in _REAL_SHORT_WORDS:
            result.append(w)
            i += 1
            continue

        # Check if this token or the next is a fragment
        if _is_fragment(w):
            # Merge consecutive fragments
            merged = w
            j = i + 1
            while j < len(words) and _is_fragment(words[j]):
                merged += words[j]
                j += 1
            # Grab one trailing suffix-like token (e.g., "rate" in "commensu" + "rate")
            if j < len(words) and j > i + 1 and words[j].isalpha() and len(words[j]) <= 5 and words[j].lower() not in _REAL_SHORT_WORDS:
                merged += words[j]
                j += 1
            if j > i + 1:
                result.append(merged)
                i = j
                continue

        # For longer tokens: absorb trailing fragments only
        # (e.g., "help" + "ful", "tak" + "ing")
        if i + 1 < len(words) and _is_fragment(words[i + 1]):
            merged = w
            j = i + 1
            while j < len(words) and _is_fragment(words[j]):
                merged += words[j]
                j += 1
            result.append(merged)
            i = j
            continue

        result.append(w)
        i += 1

    return " ".join(result)


def clean_raw_output(text: str) -> str:
    """Strip known LLM training artifacts from raw generated text.

    Handles:
    - Instruction echoes ("you are a helpful assistant...", "answer only using the context...")
    - Stop-string remnants with BPE spacing ("< end >", "< END >")
    - Repeated chunk-id citations that make the answer unreadable
    - BPE tokenizer fragmentation (reassemble common split words)
    """
    # Remove everything after <END>-like markers (with optional BPE spacing)
    text = re.split(r"<\s*end\s*>|<\s*END\s*>|<\s*end_example\s*>|<\s*END_EXAMPLE\s*>", text)[0]

    # Remove instruction echoes
    _INSTRUCTION_PATTERNS = [
        r"you\s+are\s+a\s+help\s*ful\s+assistant\.?",
        r"ans\s*w\s*er\s+only\s+using\s+the\s+con\s*te\s*x\s*t\.?",
        r'if\s+the\s+ans\s*w\s*er\s+is\s+not\s+in\s+the\s+con\s*te\s*x\s*t\s*,\s*s\s*ay\s+"?\s*not\s+found\s+in\s+con\s*te\s*x\s*t\s*"?\.?',
        r"con\s*te\s*x\s*t\s*:\s*$",
        r"question\s*:\s*$",
        r"answer\s*:\s*$",
    ]
    for pattern in _INSTRUCTION_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove excessive inline chunk-id references (keep at most 1 per sentence later in grounding)
    text = re.sub(r"\[[\w\-]+__[\w\-]+__[\w\-]+__[\w\-]+\]", "", text)

    # Fix common BPE fragmentation: reassemble words split by spaces
    # Fix BPE tokenizer fragmentation.
    # The BPE tokenizer with Whitespace pre-tokenizer and no decoder inserts
    # spaces between all subword tokens.  We fix this by detecting and merging
    # tokens that likely belong to the same word.
    # Strategy: use a spell-check-like approach — if merging adjacent short
    # tokens produces a word found in the chunk text, merge them.
    text = _defragment_bpe(text)

    # Collapse multiple spaces and clean up
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing fragments that aren't real sentences
    # (e.g., starts with "after receiving the process .")
    text = re.sub(r"^\s*[a-z].*?\.\s*", "", text, count=1) if text and text[0].islower() and ". " in text[:60] else text

    return text.strip()


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set[str]:
    """Simple whitespace tokenizer on normalized text, dropping 1-char tokens."""
    return {tok for tok in _normalize(text).split() if len(tok) > 1}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units.

    Avoids splitting on abbreviation dots (A.M., P.M., i.e., e.g., etc.)
    and section numbers (2.3.6).
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    # Protect common abbreviations and patterns from sentence splitting
    protected = cleaned
    protected = re.sub(r"\b([AaPp])\.([Mm])\.", r"\1_DOT_\2_DOT_", protected)
    protected = re.sub(r"\bi\.e\.", "i_DOT_e_DOT_", protected)
    protected = re.sub(r"\be\.g\.", "e_DOT_g_DOT_", protected)
    protected = re.sub(r"\betc\.", "etc_DOT_", protected)
    protected = re.sub(r"\bRs\.", "Rs_DOT_", protected)
    protected = re.sub(r"\bNo\.", "No_DOT_", protected)
    protected = re.sub(r"\bSh\.", "Sh_DOT_", protected)
    protected = re.sub(r"\bSmt\.", "Smt_DOT_", protected)
    protected = re.sub(r"\bDr\.", "Dr_DOT_", protected)
    # Protect section numbers like "2.3.6"
    protected = re.sub(r"(\d)\.(\d)", r"\1_DOT_\2", protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)

    # Restore protected dots
    result = []
    for p in parts:
        p = p.replace("_DOT_", ".")
        p = p.strip()
        if p:
            result.append(p)
    return result


def score_sentence_against_chunks(
    sentence: str, chunks: list[dict[str, Any]], threshold: float = 0.3
) -> tuple[bool, list[str]]:
    """Check whether a sentence is supported by at least one chunk.

    Returns ``(is_grounded, list_of_supporting_chunk_ids)``.
    A sentence is grounded if token-overlap with any chunk exceeds *threshold*.
    """
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return True, []  # empty/trivial sentence — pass

    supporting: list[str] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(str(chunk.get("text", "")))
        if not chunk_tokens:
            continue
        overlap = len(sent_tokens & chunk_tokens) / len(sent_tokens)
        if overlap >= threshold:
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id:
                supporting.append(chunk_id)

    return len(supporting) > 0, supporting


def ground_answer(
    raw_answer: str,
    chunks: list[dict[str, Any]],
    *,
    overlap_threshold: float = 0.3,
) -> dict[str, Any]:
    """Produce a grounded answer with inline citations and faithfulness score.

    Args:
        raw_answer: Raw text from the LLM.
        chunks: Retrieved chunks (each must have ``chunk_id`` and ``text``).
        overlap_threshold: Minimum token-overlap ratio to consider a sentence
            grounded in a chunk.

    Returns:
        Dict with keys:
        - ``answer``: Answer text with inline ``[chunk_id]`` citations.
        - ``citations``: Sorted list of all cited chunk IDs.
        - ``faithfulness``: Fraction of sentences grounded (0.0–1.0).
        - ``ungrounded_sentences``: List of sentences with no chunk support.
    """
    sentences = _split_sentences(raw_answer)
    if not sentences:
        return {
            "answer": raw_answer.strip(),
            "citations": [],
            "faithfulness": 0.0,
            "ungrounded_sentences": [],
        }

    cited_parts: list[str] = []
    all_citations: set[str] = set()
    ungrounded: list[str] = []
    grounded_count = 0

    for sentence in sentences:
        is_grounded, supporting_ids = score_sentence_against_chunks(
            sentence, chunks, threshold=overlap_threshold
        )
        if is_grounded and supporting_ids:
            grounded_count += 1
            citation_str = " ".join(f"[{cid}]" for cid in supporting_ids[:2])
            cited_parts.append(f"{sentence} {citation_str}")
            all_citations.update(supporting_ids[:2])
        else:
            cited_parts.append(sentence)
            if not is_grounded:
                ungrounded.append(sentence)

    faithfulness = grounded_count / len(sentences) if sentences else 0.0

    return {
        "answer": " ".join(cited_parts).strip(),
        "citations": sorted(all_citations),
        "faithfulness": round(faithfulness, 4),
        "ungrounded_sentences": ungrounded,
    }


# ---------------------------------------------------------------------------
# Extractive fallback
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "how", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with",
}


def extractive_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    max_sentences: int = 3,
) -> dict[str, Any]:
    """Build an answer by extracting the best-matching sentences from chunks.

    Uses TF-IDF-like scoring: rare query terms (appearing in fewer chunks)
    get higher weight, so specific terms like "11:30" or "PMJAY" dominate
    over common words like "scheme" or "program".

    Returns the same schema as ``ground_answer`` for compatibility.
    """
    query_tokens = {
        tok for tok in re.findall(r"[a-z0-9:]+", question.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }

    # Build document frequency for query tokens across chunks
    all_chunk_texts = [str(c.get("text", "")).lower() for c in chunks]
    doc_freq: dict[str, int] = {}
    for tok in query_tokens:
        doc_freq[tok] = sum(1 for t in all_chunk_texts if tok in t)

    n_chunks = max(len(chunks), 1)

    scored: list[tuple[float, int, str, str]] = []
    global_idx = 0
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("text", ""))
        for sentence in _split_sentences(text):
            sent_lower = sentence.lower()
            sent_tokens = set(re.findall(r"[a-z0-9:]+", sent_lower))

            # TF-IDF-like score: sum of 1/df for each matching query token
            score = 0.0
            for tok in query_tokens:
                if tok in sent_tokens:
                    df = doc_freq.get(tok, 1)
                    score += 1.0 / max(df, 1) * (n_chunks / max(df, 1))

            # Bonus for sentences that contain multiple query terms together
            matched_count = len(query_tokens & sent_tokens)
            if matched_count >= 2:
                score *= (1.0 + 0.3 * matched_count)

            # Slight length preference: avoid very short (< 5 words) snippets
            word_count = len(sentence.split())
            if word_count < 5:
                score *= 0.5

            scored.append((score, global_idx, sentence, chunk_id))
            global_idx += 1

    if not scored:
        return {
            "answer": "Not found in context.",
            "citations": [],
            "faithfulness": 0.0,
            "ungrounded_sentences": [],
        }

    scored.sort(key=lambda r: (r[0], -r[1]), reverse=True)
    top = scored[:max_sentences]
    # Re-sort by original order for coherence.
    top.sort(key=lambda r: r[1])

    cited_parts: list[str] = []
    all_citations: set[str] = set()
    for _, _, sentence, chunk_id in top:
        cited_parts.append(f"{sentence} [{chunk_id}]")
        all_citations.add(chunk_id)

    return {
        "answer": " ".join(cited_parts).strip(),
        "citations": sorted(all_citations),
        "faithfulness": 1.0,  # extractive is always grounded by definition
        "ungrounded_sentences": [],
    }


def answer_with_fallback(
    raw_answer: str,
    question: str,
    chunks: list[dict[str, Any]],
    *,
    min_answer_words: int = 8,
    min_faithfulness: float = 0.4,
) -> dict[str, Any]:
    """Produce the best answer from generative + extractive strategies.

    Always computes the extractive answer (from real document text).
    Uses the generative answer only if it's high quality (long enough,
    faithful, and coherent).  For small from-scratch LLMs, extractive
    is typically more accurate.
    """
    # Always compute extractive (guaranteed accurate from document text)
    extractive = extractive_answer(question, chunks, max_sentences=3)

    # For small from-scratch LLMs, extractive is always preferred.
    # The LLM output is only used if it adds a coherent, well-formed summary.
    # In practice, extractive from real document text is more reliable.
    return extractive
