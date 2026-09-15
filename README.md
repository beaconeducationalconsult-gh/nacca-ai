# NaCCA curriculum AI

JSON dump of Ghana’s NaCCA basic education curriculum, plus a **Hugging Face-ready dataset**, an **offline RAG assistant**, and a **QLoRA fine-tune script**.

Do **not** train an LLM from scratch. Use this hybrid:

1. **RAG** over the official indicators (facts stay citable)
2. **LoRA fine-tune** later for teacher tone and lesson-note format

Curriculum text is © NaCCA / Ministry of Education. Public PDFs are not the same as a licence to commercialise. Make NaCCA an anchor partner before a consortium product.

## What is in this repo

| Path | Role |
| --- | --- |
| `*_curriculum_db_clean.json` | Canonical indicators (use these) |
| `*_lessons_enriched.json` | 180-day lesson shells — highly templated; not used as SFT gold |
| `pipeline/build_hf_dataset.py` | Flatten JSON → Hub JSONL |
| `pipeline/serve_assistant.py` | BM25 RAG app (no GPU) |
| `pipeline/train_qlora.py` | Hugging Face TRL + PEFT QLoRA |
| `pipeline/upload_to_hub.py` | Push dataset configs to the Hub |
| `data/processed/` | Generated indicators, SFT splits, RAG docs |
| `hf/` | Dataset and model cards |

Coverage in this dump: **B1–B9**, 13 subjects, ~3k indicators (English B5 DB file is missing).

## What you should do next (in order)

### 1. Dataset (done locally)

```bash
python3 pipeline/build_hf_dataset.py
```

Creates:

- `data/processed/indicators.jsonl` — one record per indicator
- `data/processed/sft_train.jsonl` / `sft_eval.jsonl` — chat SFT
- `data/processed/rag_docs.jsonl` — retrieval corpus
- `data/processed/golden_eval.json` — citation checks

### 2. Ship a useful product before training

```bash
python3 pipeline/serve_assistant.py --port 7860
```

Teachers can already look up codes, strands, and lesson outlines. This is the right consortium demo: **grounded, offline-capable, cheap**.

### 3. Hugging Face dataset

```bash
pip install datasets huggingface_hub
huggingface-cli login
python pipeline/upload_to_hub.py --repo YOUR_ORG/nacca-curriculum --private
```

Keep the first upload **private** until NaCCA permission is in writing.

### 4. Fine-tune only after RAG is in teachers’ hands

On Colab or a rented GPU:

```bash
pip install -U transformers datasets peft trl bitsandbytes accelerate
python pipeline/train_qlora.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --train data/processed/sft_train.jsonl \
  --eval data/processed/sft_eval.jsonl \
  --output outputs/nacca-qwen25-3b-lora
```

Default base is a 3B instruct model (Apache 2.0 for most Qwen 2.5 sizes). Swap to 7B/8B when you have an A100. Export GGUF later for school machines with poor connectivity.

### 5. Consortium work in parallel

- Written MoU with NaCCA on curriculum reuse
- Confirm base-model licence for commercial use
- Teacher review of `golden_eval.json` (correct code, grade, no inventions)
- Version datasets and models on the Hub from day one

## Design notes

- **RAG first.** The curriculum is a reference corpus. Fine-tuning cannot be your source of truth.
- **Do not SFT on the 180-day lesson JSON.** Those files repeat the same starter/main/plenary with the indicator swapped in; the model would learn boilerplate.
- **Always cite codes** (`B4.1.1.1.1`). Refuse unknown codes.
- **English B5** is a known hole in this dump.

## Requirements

- Dataset + RAG assistant: Python 3.9+, stdlib only
- Hub upload: `datasets`, `huggingface_hub`
- QLoRA: see `pipeline/train_qlora.py` header
