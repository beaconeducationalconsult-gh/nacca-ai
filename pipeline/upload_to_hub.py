#!/usr/bin/env python3
"""Upload the processed NaCCA JSONL dataset to the Hugging Face Hub.

    huggingface-cli login
    python pipeline/upload_to_hub.py --repo your-org/nacca-curriculum-indicators
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hub dataset repo, e.g. org/nacca-curriculum")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    try:
        from datasets import Features, Sequence, Value, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install: pip install datasets huggingface_hub\n" f"Original error: {exc}"
        ) from exc

    train = args.data_dir / "sft_train.jsonl"
    eval_path = args.data_dir / "sft_eval.jsonl"
    indicators = args.data_dir / "indicators.jsonl"
    if not train.exists() or not indicators.exists():
        raise SystemExit("Run python3 pipeline/build_hf_dataset.py first.")

    sft = load_dataset(
        "json",
        data_files={"train": str(train), "validation": str(eval_path)},
    )
    sft.push_to_hub(args.repo, config_name="sft", private=args.private)

    inds = load_dataset("json", data_files={"train": str(indicators)})
    inds.push_to_hub(args.repo, config_name="indicators", private=args.private)
    print(f"Pushed indicators + SFT configs to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
