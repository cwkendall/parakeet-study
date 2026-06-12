"""
Transplant encoder weights from a trained CTC model into a fresh RNN-T model.

Usage:
    python init_from_ctc.py \
        --ctc_checkpoint ../04-train-fastconformer-ctc/lightning_logs/version_0/checkpoints/last.ckpt \
        --rnnt_config conf/fastconformer_rnnt_small.yaml \
        --output rnnt_init.nemo
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctc_checkpoint", required=True)
    p.add_argument("--rnnt_config", required=True)
    p.add_argument("--tokenizer_dir", required=True,
                   help="Path to SentencePiece tokenizer directory")
    p.add_argument("--output", required=True,
                   help="Path to save the initialised .nemo file")
    args = p.parse_args()

    # Load the trained CTC model
    print(f"Loading CTC checkpoint from {args.ctc_checkpoint}")
    ctc = EncDecCTCModelBPE.load_from_checkpoint(args.ctc_checkpoint, map_location="cpu")

    # Load the RNN-T config and patch the tokenizer path
    cfg = OmegaConf.load(args.rnnt_config)
    cfg.model.tokenizer.dir = args.tokenizer_dir

    # Build a fresh RNN-T model
    rnnt = EncDecRNNTBPEModel(cfg=cfg.model)

    # Copy encoder weights
    ctc_sd = ctc.encoder.state_dict()
    missing, unexpected = rnnt.encoder.load_state_dict(ctc_sd, strict=False)
    print(f"Copied {len(ctc_sd) - len(unexpected)} encoder tensors. "
          f"missing={len(missing)}, unexpected={len(unexpected)}.")
    if missing or unexpected:
        print("  missing in target  :", missing[:5], "..." if len(missing) > 5 else "")
        print("  unexpected from src:", unexpected[:5], "..." if len(unexpected) > 5 else "")

    # Save
    rnnt.save_to(args.output)
    print(f"Saved initialised RNN-T model to {args.output}")
    print("Run training with:")
    print(f"  python train.py ... +init_from_nemo_model={args.output}")


if __name__ == "__main__":
    main()
