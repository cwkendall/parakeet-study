"""
Train a SentencePiece BPE tokenizer on the LibriSpeech training transcripts.

Usage:
    python train_tokenizer.py \
        --manifest data/train-clean-100/train_clean_100.json \
        --vocab_size 1024 \
        --output_dir tokenizer
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import sentencepiece as spm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--output_dir", default="tokenizer")
    p.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # SentencePiece reads a plain text file (one sentence per line). Extract
    # transcripts from the NeMo manifest into a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        text_path = f.name
        n = 0
        with open(args.manifest) as m:
            for line in m:
                row = json.loads(line)
                f.write(row["text"].strip().lower() + "\n")
                n += 1
        print(f"Wrote {n} transcripts to {text_path}")

    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=os.path.join(args.output_dir, "tokenizer"),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,  # CTC: no SOS/EOS, blank handled separately
        unk_piece="<unk>",
    )
    print(f"Tokenizer model: {args.output_dir}/tokenizer.model")
    print(f"Vocabulary:       {args.output_dir}/tokenizer.vocab")

    # Smoke test
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.output_dir, "tokenizer.model"))
    for s in ["the quick brown fox", "speech recognition is hard", "parakeet"]:
        print(f"  {s!r:40s} -> {sp.encode(s, out_type=str)}")

    os.unlink(text_path)


if __name__ == "__main__":
    main()
