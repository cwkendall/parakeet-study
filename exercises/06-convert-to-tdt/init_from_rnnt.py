"""
Transplant encoder + prednet + joint-pre weights from a trained RNN-T model
into a fresh TDT model. The joint's output layer is reshaped: the first V+1
rows (token logits) are copied; the trailing 5 rows (duration logits) are
left at random init.

Usage:
    python init_from_rnnt.py \
        --rnnt_checkpoint ../05-convert-to-rnnt/lightning_logs/.../last.ckpt \
        --tdt_config conf/fastconformer_tdt_small.yaml \
        --tokenizer_dir ../04-train-fastconformer-ctc/tokenizer \
        --output tdt_init.nemo
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rnnt_checkpoint", required=True)
    p.add_argument("--tdt_config", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Loading RNN-T checkpoint from {args.rnnt_checkpoint}")
    rnnt = EncDecRNNTBPEModel.load_from_checkpoint(args.rnnt_checkpoint, map_location="cpu")

    cfg = OmegaConf.load(args.tdt_config)
    cfg.model.tokenizer.dir = args.tokenizer_dir

    print("Building fresh TDT model from config")
    tdt = EncDecRNNTBPEModel(cfg=cfg.model)

    # --- Encoder: identical -> direct copy
    missing, unexpected = tdt.encoder.load_state_dict(rnnt.encoder.state_dict(), strict=False)
    print(f"Encoder:  copied {len(rnnt.encoder.state_dict()) - len(unexpected)} tensors  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")

    # --- Prednet: identical -> direct copy
    pred_sd = {k: v for k, v in rnnt.decoder.state_dict().items()}
    missing, unexpected = tdt.decoder.load_state_dict(pred_sd, strict=False)
    print(f"Prednet:  copied {len(pred_sd) - len(unexpected)} tensors  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")

    # --- Joint network: pre-projection identical; out-projection partial copy
    rnnt_joint_sd = rnnt.joint.state_dict()
    tdt_joint_sd = tdt.joint.state_dict()

    # NeMo's RNNTJoint stores the final linear layer as joint_net.<something> 
    # depending on version; do a robust per-tensor reconciliation.
    n_copied = n_partial = 0
    for k, v in rnnt_joint_sd.items():
        if k not in tdt_joint_sd:
            continue
        target = tdt_joint_sd[k]
        if target.shape == v.shape:
            target.copy_(v)
            n_copied += 1
        elif target.dim() == v.dim() and target.shape[1:] == v.shape[1:] and target.shape[0] > v.shape[0]:
            # Output projection: target has more rows (the duration outputs).
            target[: v.shape[0]] = v
            n_partial += 1
            print(f"  joint partial copy: {k}  {v.shape} -> {target.shape}")
        elif target.dim() == 1 and target.shape[0] > v.shape[0]:
            target[: v.shape[0]] = v
            n_partial += 1
            print(f"  joint partial copy (bias): {k}  {v.shape} -> {target.shape}")
        else:
            print(f"  SKIPPED (shape mismatch): {k}  ours {target.shape}  theirs {v.shape}")

    tdt.joint.load_state_dict(tdt_joint_sd)
    print(f"Joint:    {n_copied} exact + {n_partial} partial-copy tensors")
    print(f"          (duration-output rows initialised at random)")

    tdt.save_to(args.output)
    print(f"\nSaved initialised TDT model to {args.output}")


if __name__ == "__main__":
    main()
