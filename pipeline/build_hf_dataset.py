#!/usr/bin/env python3
"""Flatten NaCCA JSON into Hugging Face-ready JSONL (indicators + SFT pairs).

Usage:
    python3 pipeline/build_hf_dataset.py
    python3 pipeline/build_hf_dataset.py --root . --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SYSTEM_PROMPT, display_grade, grade_band, parse_source_file, subject_title

CODE_RE = re.compile(r"\bB\d(?:\.\d+){2,4}\b", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_list(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;|]", value)
    return [p.strip() for p in parts if p.strip()]


def is_eval_split(indicator_code: str, eval_ratio: float = 0.08) -> bool:
    digest = hashlib.sha256(indicator_code.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < eval_ratio


def collect_indicators(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for path in sorted(root.glob("*_curriculum_db_clean.json")):
        parsed = parse_source_file(path.name)
        if not parsed:
            continue
        subject_slug, grade = parsed
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        for code, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            key = (subject_slug, str(code).strip())
            if key in seen:
                continue
            seen.add(key)
            records.append(normalize_indicator(subject_slug, grade, str(code), raw, path.name))
    records.sort(key=lambda r: (r["subject_slug"], r["grade"], r["indicator_code"]))
    return records


def normalize_indicator(
    subject_slug: str, grade: int, code: str, raw: dict[str, Any], source_file: str
) -> dict[str, Any]:
    strand = clean_text(raw.get("strand"))
    sub_strand = clean_text(raw.get("sub_strand"))
    cs_code = clean_text(raw.get("cs_code")) or infer_cs_code(code)
    cs_desc = clean_text(raw.get("cs_desc"))
    ind_desc = clean_text(raw.get("ind_desc"))
    competencies = clean_text(raw.get("competencies"))
    resources = clean_text(raw.get("resources"))
    assessment = clean_text(raw.get("assessment"))
    keywords = clean_text(raw.get("keywords"))
    return {
        "indicator_code": code.strip(),
        "subject_slug": subject_slug,
        "subject": subject_title(subject_slug),
        "grade": grade,
        "grade_label": display_grade(grade),
        "grade_band": grade_band(grade),
        "strand": strand,
        "sub_strand": sub_strand,
        "content_standard_code": cs_code,
        "content_standard": cs_desc,
        "indicator": ind_desc,
        "competencies": competencies,
        "competency_list": split_list(competencies),
        "resources": resources,
        "assessment": assessment,
        "keywords": keywords,
        "source_file": source_file,
        "text": indicator_search_text(
            subject_title(subject_slug), grade, code, strand, sub_strand, cs_code, cs_desc, ind_desc
        ),
    }


def infer_cs_code(indicator_code: str) -> str:
    parts = indicator_code.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:4])
    return indicator_code


def indicator_search_text(
    subject: str,
    grade: int,
    code: str,
    strand: str,
    sub_strand: str,
    cs_code: str,
    cs_desc: str,
    ind_desc: str,
) -> str:
    return (
        f"{subject} {display_grade(grade)} indicator {code}. "
        f"Strand: {strand}. Sub-strand: {sub_strand}. "
        f"Content standard {cs_code}: {cs_desc}. "
        f"Learners should be able to: {ind_desc}."
    )


def format_indicator_card(rec: dict[str, Any]) -> str:
    lines = [
        f"Subject: {rec['subject']} ({rec['grade_label']}, {rec['grade_band']})",
        f"Indicator code: {rec['indicator_code']}",
        f"Strand: {rec['strand'] or '—'}",
        f"Sub-strand: {rec['sub_strand'] or '—'}",
        f"Content standard ({rec['content_standard_code']}): {rec['content_standard'] or '—'}",
        f"Indicator: {rec['indicator'] or '—'}",
    ]
    if rec["competencies"]:
        lines.append(f"Core competencies: {rec['competencies']}")
    if rec["resources"]:
        lines.append(f"Suggested resources: {rec['resources']}")
    if rec["assessment"]:
        lines.append(f"Assessment: {rec['assessment']}")
    lines.append(
        "Source: NaCCA basic education curriculum records in this dataset. "
        "Do not treat this as a substitute for the official PDF."
    )
    return "\n".join(lines)


def format_lesson_outline(rec: dict[str, Any]) -> str:
    topic = rec["indicator"] or rec["content_standard"] or rec["indicator_code"]
    return "\n".join(
        [
            f"NaCCA-aligned lesson outline — {rec['subject']} {rec['grade_label']}",
            f"Indicator: {rec['indicator_code']} — {rec['indicator']}",
            f"Content standard: {rec['content_standard_code']} — {rec['content_standard']}",
            f"Strand / sub-strand: {rec['strand']} / {rec['sub_strand']}",
            "",
            f"Performance indicator: By the end of the lesson, learners will be able to {topic[0].lower() + topic[1:] if topic else 'meet the indicator'}.",
            f"Core competencies: {rec['competencies'] or 'NaCCA core competencies as specified in the curriculum.'}",
            f"Resources: {rec['resources'] or 'NaCCA-approved textbook; TLMs; local/community resources.'}",
            "",
            "Starter (5–8 min): Activate relevant previous knowledge; introduce the indicator in child-friendly language; write key words on the board.",
            "Main (concrete → pictorial → abstract):",
            f"  1. Teacher models the skill in {rec['indicator']}.",
            "  2. Guided pair/group practice with teacher support for struggling learners.",
            "  3. Independent or pictorial practice linked to the sub-strand.",
            "  4. Selected learners share; class agrees the rule or success criteria.",
            "Plenary: Oral check against the performance indicator; one learner restates the idea; short homework tied to the same indicator.",
            f"Assessment: {rec['assessment'] or 'Observation, oral questions, class exercise, and SBA evidence aligned to the indicator.'}",
            "",
            "Cite this indicator in schemes of work and lesson notes. Do not invent extra NaCCA codes.",
        ]
    )


def sft_example(user: str, assistant: str, rec: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "task": task,
        "split_hint": rec["indicator_code"],
        "subject": rec["subject"],
        "subject_slug": rec["subject_slug"],
        "grade": rec["grade"],
        "indicator_code": rec["indicator_code"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def generate_sft_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_subject_grade: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_cs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_subject_grade[(rec["subject_slug"], rec["grade"])].append(rec)
        by_cs[(rec["subject_slug"], rec["content_standard_code"])].append(rec)

    examples: list[dict[str, Any]] = []
    for rec in records:
        code = rec["indicator_code"]
        subj = rec["subject"]
        gl = rec["grade_label"]
        card = format_indicator_card(rec)

        examples.append(
            sft_example(
                f"What does NaCCA indicator {code} say in {subj} {gl}?",
                card,
                rec,
                "lookup_code",
            )
        )
        examples.append(
            sft_example(
                f"Explain {code} for a {gl} {subj} teacher writing lesson notes.",
                card,
                rec,
                "teacher_explain",
            )
        )
        examples.append(
            sft_example(
                f"Design a lesson aligned to NaCCA {subj} {gl} indicator {code}: {rec['indicator']}",
                format_lesson_outline(rec),
                rec,
                "lesson_outline",
            )
        )
        examples.append(
            sft_example(
                f"How should I assess {gl} learners on {code} ({subj})?",
                (
                    f"Assess {code} against the indicator: {rec['indicator']}\n"
                    f"Suggested modes: {rec['assessment'] or 'class exercise, oral questions, practical performance, SBA.'}\n"
                    "Collect evidence that the learner can do the indicator, not a generic topic test. "
                    "Record the indicator code on the assessment item."
                ),
                rec,
                "assessment",
            )
        )

        siblings = by_cs[(rec["subject_slug"], rec["content_standard_code"])]
        if rec is siblings[0]:
            listing = "\n".join(f"- {s['indicator_code']}: {s['indicator']}" for s in siblings)
            examples.append(
                sft_example(
                    f"List the NaCCA indicators under content standard {rec['content_standard_code']} "
                    f"in {subj} {gl}.",
                    (
                        f"Content standard {rec['content_standard_code']}: {rec['content_standard']}\n"
                        f"Subject/grade: {subj} {gl}\nIndicators:\n{listing}"
                    ),
                    rec,
                    "list_content_standard",
                )
            )

    for (slug, grade), group in by_subject_grade.items():
        rec0 = group[0]
        strands = []
        seen_strand = set()
        for rec in group:
            key = rec["strand"]
            if key and key not in seen_strand:
                seen_strand.add(key)
                n = sum(1 for r in group if r["strand"] == key)
                strands.append(f"- {key} ({n} indicators)")
        examples.append(
            sft_example(
                f"Which strands are in the NaCCA {rec0['subject']} curriculum for {rec0['grade_label']}?",
                (
                    f"{rec0['subject']} {rec0['grade_label']} ({rec0['grade_band']}) has "
                    f"{len(group)} indicators across {len(strands)} strand(s):\n"
                    + "\n".join(strands)
                ),
                rec0,
                "strands_overview",
            )
        )

    # Grounded refusal: a plausible-looking code that is not in this dump.
    fake = next(r for r in records if r["subject_slug"] == "mathematics" and r["grade"] == 4)
    examples.append(
        sft_example(
            "What does NaCCA indicator B4.9.9.9.9 cover in Mathematics?",
            (
                "B4.9.9.9.9 is not in this NaCCA Mathematics B4 indicator dump. "
                "I will not invent a learning outcome. Please check the official "
                "NaCCA PDF or ask using a real code such as "
                f"{fake['indicator_code']}."
            ),
            fake,
            "refuse_unknown_code",
        )
    )
    return examples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build(root: Path, out: Path) -> dict[str, Any]:
    records = collect_indicators(root)
    if not records:
        raise SystemExit(f"No curriculum_db_clean JSON found under {root}")

    examples = generate_sft_pairs(records)
    train, eval_rows = [], []
    for ex in examples:
        (eval_rows if is_eval_split(ex["split_hint"]) else train).append(ex)

    rag_docs = [
        {
            "doc_id": f"{r['subject_slug']}:{r['indicator_code']}",
            "indicator_code": r["indicator_code"],
            "subject": r["subject"],
            "subject_slug": r["subject_slug"],
            "grade": r["grade"],
            "grade_label": r["grade_label"],
            "text": r["text"],
            "record": r,
        }
        for r in records
    ]

    stats = {
        "n_indicators": len(records),
        "n_sft_train": len(train),
        "n_sft_eval": len(eval_rows),
        "n_rag_docs": len(rag_docs),
        "by_subject": dict(Counter(r["subject"] for r in records)),
        "by_grade": dict(sorted(Counter(r["grade_label"] for r in records).items())),
        "known_gaps": [
            "english-language B5 curriculum_db_clean.json is missing from this dump "
            "(summary and lessons exist).",
            "Several B1 subjects exist only as ungraded *_curriculum_db_clean.json files.",
            "OWOP coverage in this dump is B1 and B4–B6 only.",
        ],
    }

    write_jsonl(out / "indicators.jsonl", records)
    write_jsonl(out / "sft_train.jsonl", train)
    write_jsonl(out / "sft_eval.jsonl", eval_rows)
    write_jsonl(out / "rag_docs.jsonl", rag_docs)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    golden = []
    for rec in records:
        if rec["indicator_code"].endswith(".1") and rec["grade"] in {2, 4, 7}:
            golden.append(
                {
                    "question": f"What is {rec['indicator_code']} in {rec['subject']} {rec['grade_label']}?",
                    "must_cite": rec["indicator_code"],
                    "subject": rec["subject"],
                    "grade": rec["grade"],
                }
            )
        if len(golden) >= 120:
            break
    (out / "golden_eval.json").write_text(json.dumps(golden, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
    )
    args = parser.parse_args()
    stats = build(args.root, args.out)
    print(json.dumps(stats, indent=2))
    print(f"Wrote dataset to {args.out}")


if __name__ == "__main__":
    main()
