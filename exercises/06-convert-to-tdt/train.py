"""Train the TDT model. Same shape as exercise 5's train.py."""
import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="fastconformer_tdt_small")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    init_from = cfg.get("init_from_nemo_model", None)
    if init_from:
        logging.info("Initialising weights from %s", init_from)
        asr_model = EncDecRNNTBPEModel.restore_from(init_from, override_config_path=cfg)
        asr_model.set_trainer(trainer)

    n_params = sum(p.numel() for p in asr_model.parameters())
    print(f"Total parameters: {n_params/1e6:.2f} M")
    print(f"Joint num_extra_outputs (= |D|): {asr_model.joint.num_extra_outputs}")
    print(f"TDT durations: {cfg.model.loss.tdt_kwargs.durations}")
    print(f"sigma = {cfg.model.loss.tdt_kwargs.sigma}, omega = {cfg.model.loss.tdt_kwargs.omega}")

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
