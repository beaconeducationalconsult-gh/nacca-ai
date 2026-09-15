"""Canonical field names and filename → subject/grade mapping."""

from __future__ import annotations

import re
from typing import Optional

SUBJECT_TITLES = {
    "career-technology": "Career Technology",
    "computing": "Computing",
    "creative-arts": "Creative Arts",
    "creative-arts-design": "Creative Arts and Design",
    "english-language": "English Language",
    "french": "French",
    "ghanaian-language": "Ghanaian Language",
    "history": "History",
    "mathematics": "Mathematics",
    "owop": "Our World Our People",
    "rme": "Religious and Moral Education",
    "science": "Science",
    "social-studies": "Social Studies",
}

# Ungraded *_curriculum_db_clean.json files in this dump are Basic 1 extracts.
B1_GENERIC_FILES = {
    "creative_arts_curriculum_db_clean.json": "creative-arts",
    "english_curriculum_db_clean.json": "english-language",
    "ghanaian_language_curriculum_db_clean.json": "ghanaian-language",
    "history_curriculum_db_clean.json": "history",
    "math_curriculum_db_clean.json": "mathematics",
    "owop_curriculum_db_clean.json": "owop",
    "rme_curriculum_db_clean.json": "rme",
    "science_curriculum_db_clean.json": "science",
}

GRADED_DB_RE = re.compile(
    r"^(?P<subject>.+)_B(?P<grade>\d)_curriculum_db_clean\.json$"
)

SYSTEM_PROMPT = (
    "You are a NaCCA curriculum assistant for Ghanaian basic education "
    "(Basic 1–9). Answer only from the official NaCCA curriculum records "
    "provided. Always cite indicator codes (for example B4.1.1.1.1). "
    "If an indicator, strand, or subject is not in the records, say so "
    "clearly — never invent NaCCA codes or learning outcomes."
)


def grade_band(grade: int) -> str:
    if grade <= 3:
        return "lower-primary"
    if grade <= 6:
        return "upper-primary"
    return "jhs-ccp"


def display_grade(grade: int) -> str:
    return f"B{grade}"


def parse_source_file(filename: str) -> Optional[tuple[str, int]]:
    """Return (subject_slug, grade) or None if the file is not a curriculum DB."""
    match = GRADED_DB_RE.match(filename)
    if match:
        return match.group("subject"), int(match.group("grade"))
    if filename in B1_GENERIC_FILES:
        return B1_GENERIC_FILES[filename], 1
    return None


def subject_title(slug: str) -> str:
    return SUBJECT_TITLES.get(slug, slug.replace("-", " ").title())
