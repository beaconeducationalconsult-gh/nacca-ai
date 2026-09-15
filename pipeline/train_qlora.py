#!/usr/bin/env python3
"""QLoRA supervised fine-tune on the NaCCA SFT JSONL (Hugging Face TRL + PEFT).

This script is meant to run on a GPU box or Colab — not in this CPU sandbox.

Example (Colab / rented A100):

    pip install -U transformers datasets peft trl bitsandbytes accelerate
    python pipeline/train_qlora.py \\
        --train data/processed/sft_train.jsonl \\
        --eval data/processed/sft_eval.jsonl \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --output outputs/nacca-qwen25-3b-lora

Then merge or push the adapter:

    python pipeline/train_qlora.py --push --hub-repo your-org/nacca-curriculum-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/processed/sft_eval.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/nacca-qlora"))
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--hub-repo", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training extras. Install with:\n"
            "  pip install -U transformers datasets peft trl bitsandbytes accelerate\n"
            f"Original error: {exc}"
        ) from exc

    if not args.train.exists():
        raise SystemExit(f"Missing {args.train}. Run python3 pipeline/build_hf_dataset.py first.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    data_files = {"train": str(args.train)}
    if args.eval.exists():
        data_files["eval"] = str(args.eval)
    ds = load_dataset("json", data_files=data_files)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    sft_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if "eval" in ds else "no",
        bf16=torch.cuda.is_available(),
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    if args.push:
        repo = args.hub_repo or args.output.name
        trainer.model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)
        print(f"Pushed adapter to {repo}")
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
