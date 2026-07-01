"""
Catalog access layer.

load_catalog()      – returns the full list of assessment records.
find_by_name(name)  – exact-then-fuzzy lookup so slight LLM paraphrases
                      (e.g. "Java 8 New" vs "Java 8 (New)") still resolve.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _catalog_by_exact_name() -> dict[str, dict]:
    """Exact lower-stripped lookup."""
    return {rec["name"].strip().lower(): rec for rec in load_catalog()}


@lru_cache(maxsize=1)
def _catalog_by_id() -> dict[str, dict]:
    return {rec["id"].strip().lower(): rec for rec in load_catalog()}


def _normalise(s: str) -> str:
    """Strip punctuation/spaces for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=1)
def _catalog_by_normalised_name() -> dict[str, dict]:
    return {_normalise(rec["name"]): rec for rec in load_catalog()}


def find_by_name(name: str) -> dict | None:
    """
    Try three strategies in order:
    1. Exact name match (case-insensitive, stripped).
    2. ID match.
    3. Normalised name match (removes punctuation and spaces).
    Returns None if nothing matches.
    """
    if not name or not name.strip():
        return None

    key = name.strip().lower()

    # 1. Exact
    rec = _catalog_by_exact_name().get(key)
    if rec:
        return rec

    # 2. ID
    rec = _catalog_by_id().get(key)
    if rec:
        return rec

    # 3. Normalised
    norm = _normalise(name)
    rec = _catalog_by_normalised_name().get(norm)
    if rec:
        return rec

    # 4. Substring: the LLM might return a truncated name
    norm_lower = norm
    for norm_key, candidate in _catalog_by_normalised_name().items():
        if norm_lower in norm_key or norm_key in norm_lower:
            return candidate

    return None
