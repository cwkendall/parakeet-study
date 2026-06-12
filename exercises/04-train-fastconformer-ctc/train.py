"""
Train the small FastConformer-CTC model. Wraps NeMo's standard ASR training script.

Recommended invocation:

    python train.py \
        --config-path=conf \
        --config-name=fastconformer_ctc_small \
        model.train_ds.manifest_filepath=./data/train-clean-100/train_clean_100.json \
        model.validation_ds.manifest_filepath=./data/dev-clean/dev_clean.json \
        model.tokenizer.dir=./tokenizer
"""
from __future__ import annotations

import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="fastconformer_ctc_small")
def main(cfg):
    logging.info("Hydra config:\n%s", cfg.pretty() if hasattr(cfg, "pretty") else cfg)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # Helpful header to print before training starts
    n_params = sum(p.numel() for p in asr_model.parameters())
    n_encoder = sum(p.numel() for p in asr_model.encoder.parameters())
    n_decoder = sum(p.numel() for p in asr_model.decoder.parameters())
    print(f"=== Model parameter count ===")
    print(f"  encoder: {n_encoder / 1e6:6.2f} M")
    print(f"  decoder: {n_decoder / 1e6:6.2f} M (CTC head)")
    print(f"  total:   {n_params  / 1e6:6.2f} M")

    trainer.fit(asr_model)

    # Optional: test set evaluation if test_ds.manifest_filepath was set
    if cfg.model.get("test_ds") and cfg.model.test_ds.get("manifest_filepath"):
        trainer.test(asr_model)


if __name__ == "__main__":
    main()
