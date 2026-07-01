"""
Retrieval layer.

Uses TF-IDF over a rich corpus string per assessment. The corpus string
includes the id (URL slug), name, description, test_type label, job levels,
and duration — giving the vectoriser enough signal to match natural-language
queries about roles, skills, and assessment types.

search()          – ranked TF-IDF retrieval by query string.
search_by_names() – look up assessments by partial name / slug fragments.
"""
from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import load_catalog

# ---------------------------------------------------------------------------
# Test-type labels: expand single letters to readable words so the vectoriser
# can match queries like "personality test" or "knowledge test".
# ---------------------------------------------------------------------------
_TYPE_LABELS: dict[str, str] = {
    "K": "knowledge technical skill",
    "A": "ability cognitive reasoning aptitude",
    "P": "personality behaviour questionnaire trait",
    "S": "situational judgement judgment scenario",
    "B": "biodata structured interview",
    "C": "simulation work sample exercise",
    "D": "development 360 feedback",
    "E": "engagement survey",
}

# ---------------------------------------------------------------------------
# Lazy-initialised index (built once on first use).
# ---------------------------------------------------------------------------
_vectorizer: TfidfVectorizer | None = None
_matrix = None
_records: list[dict] | None = None


def _build_index() -> None:
    global _vectorizer, _matrix, _records
    _records = load_catalog()
    corpus = [_make_corpus_string(r) for r in _records]
    _vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # bigrams help match "java developer", "sql server", etc.
        max_df=0.95,
        min_df=1,
    )
    _matrix = _vectorizer.fit_transform(corpus)


def _make_corpus_string(r: dict) -> str:
    """Build a single rich string per assessment for TF-IDF indexing."""
    parts = [
        r.get("name", ""),
        r.get("id", "").replace("-", " "),          # URL slug as words
        r.get("description", ""),
        _TYPE_LABELS.get(r.get("test_type") or "", ""),
        " ".join(r.get("job_level") or []),
        f"duration {r.get('duration_minutes') or ''} minutes",
    ]
    return " ".join(p for p in parts if p)


def _ensure_index() -> None:
    if _vectorizer is None:
        _build_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 10) -> list[dict]:
    """Return up to top_k records ranked by TF-IDF cosine similarity."""
    _ensure_index()
    assert _records is not None and _vectorizer is not None and _matrix is not None

    if not query.strip():
        return list(_records[:top_k])

    q_vec = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _matrix).flatten()
    ranked = sorted(zip(_records, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [rec for rec, _ in ranked[:top_k]]


def search_by_names(fragments: list[str], top_k: int = 5) -> list[dict]:
    """
    Return records whose name or id contains any of the given fragments.
    Used to ensure named assessments (e.g. "OPQ", "Verify") are always
    surfaced in compare/refine turns even if the TF-IDF score is low.
    """
    _ensure_index()
    assert _records is not None

    results: list[dict] = []
    seen: set[str] = set()

    for rec in _records:
        key = (rec.get("name", "") + " " + rec.get("id", "")).lower()
        if any(f.lower() in key for f in fragments):
            if rec["id"] not in seen:
                results.append(rec)
                seen.add(rec["id"])
        if len(results) >= top_k:
            break

    return results
