---
language:
  - en
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - ghana
  - nacca
  - education
  - lora
  - peft
---

# NaCCA curriculum LoRA (draft card)

Adapter trained with `pipeline/train_qlora.py` on `data/processed/sft_train.jsonl`.

## Intended use

Teacher-facing assistant for Ghanaian basic education: look up indicators, draft lesson outlines, and cite official NaCCA codes.

## Out of scope

- Not a student tutor for exam cheating
- Not an official NaCCA product
- Must not invent indicator codes — pair with RAG over `indicators.jsonl`

## Training

- Method: QLoRA 4-bit + LoRA (`r=16`)
- Default base: `Qwen/Qwen2.5-3B-Instruct` (Apache 2.0 for this size; confirm before commercial use)
- Data: template-grounded pairs from the JSON dump (not free-form LLM synthesis)

## Evaluation

Use `data/processed/golden_eval.json` and teacher review. Check: correct code cited, correct subject/grade, no invented indicators.
