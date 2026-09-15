---
language:
  - en
license: other
task_categories:
  - question-answering
  - text-generation
pretty_name: NaCCA Ghana Basic Education Curriculum
tags:
  - ghana
  - nacca
  - curriculum
  - education
  - rag
  - sft
size_categories:
  - 1K<n<10K
---

# NaCCA Ghana Basic Education Curriculum (indicators + SFT)

Machine-readable NaCCA (National Council for Curriculum and Assessment, Ghana) indicators for Basic 1–9, plus instruction-tuning pairs for a curriculum assistant.

**This is not an official NaCCA release.** Curriculum text is © NaCCA / Ministry of Education. Seek written permission before commercial redistribution or a public Hub upload.

## Configs

| config | file | what it is |
| --- | --- | --- |
| `indicators` | `data/processed/indicators.jsonl` | one row per indicator |
| `sft` | `data/processed/sft_{train,eval}.jsonl` | chat messages for TRL `SFTTrainer` |
| RAG docs | `data/processed/rag_docs.jsonl` | BM25/embedding passages |

## Indicator fields

`indicator_code`, `subject`, `subject_slug`, `grade`, `grade_label`, `grade_band`, `strand`, `sub_strand`, `content_standard_code`, `content_standard`, `indicator`, `competencies`, `resources`, `assessment`, `source_file`, `text`

## SFT tasks

- lookup by official code
- teacher-facing explanation
- lesson outline grounded in the indicator
- assessment guidance
- list indicators under a content standard
- strands overview for a subject/grade
- refusal to invent unknown codes

## Rebuild

```bash
python3 pipeline/build_hf_dataset.py
```

## Known gaps in this dump

- English Language B5 `curriculum_db_clean.json` is missing
- OWOP is B1 and B4–B6 only
- Several B1 subjects live in ungraded `*_curriculum_db_clean.json` files
