"""Train the small FastConformer-RNN-T model."""
import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="fastconformer_rnnt_small")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # Init from a .nemo checkpoint if specified
    init_from = cfg.get("init_from_nemo_model", None)
    if init_from:
        logging.info("Initialising weights from %s", init_from)
        asr_model = EncDecRNNTBPEModel.restore_from(init_from, override_config_path=cfg)
        asr_model.set_trainer(trainer)

    n_params = sum(p.numel() for p in asr_model.parameters())
    n_encoder = sum(p.numel() for p in asr_model.encoder.parameters())
    n_decoder = sum(p.numel() for p in asr_model.decoder.parameters())
    n_joint = sum(p.numel() for p in asr_model.joint.parameters())
    print(f"=== Model parameter count ===")
    print(f"  encoder: {n_encoder / 1e6:6.2f} M")
    print(f"  decoder: {n_decoder / 1e6:6.2f} M (prednet LSTM)")
    print(f"  joint:   {n_joint   / 1e6:6.2f} M")
    print(f"  total:   {n_params  / 1e6:6.2f} M")

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
