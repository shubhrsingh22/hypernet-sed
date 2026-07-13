"""Training entrypoint (Hydra + Lightning).

After fitting, reproduces the paper's evaluation protocol: the model is scored on
the test set across its best-10 checkpoints (by validation loss) and the mean of
the segment-wise F1 scores is reported (Section 3.4 / Table 2).
"""

import json
import logging
import os
from typing import List

import hydra
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from src.utils.evaluation import evaluate_split
from src.utils.utils import instantiate_callbacks, instantiate_loggers, log_hyperparameters

log = logging.getLogger(__name__)


def _evaluate_checkpoints(cfg, model, datamodule, trainer, ckpt_paths, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = datamodule.stats
    records = datamodule.manifest[tag if tag in datamodule.manifest else "test"]
    chunk_frames = datamodule.chunk_frames
    hop_frames = datamodule.chunk_hop_frames

    per_ckpt = []
    for path in ckpt_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        net = model.net.to(device)
        res = evaluate_split(
            net, records, stats, chunk_frames, hop_frames, device,
            threshold=cfg.get("eval_threshold", 0.5),
            median_filter_frames=cfg.get("median_filter_frames", 0),
        )
        per_ckpt.append(res)
        log.info(f"[{tag}] {os.path.basename(path)} overall F1={res['overall']['f_measure']:.2f}")

    classes = stats["classes"]
    overall_f1 = [r["overall"]["f_measure"] for r in per_ckpt]
    summary = {
        "n_checkpoints": len(per_ckpt),
        "overall_f1_mean": float(sum(overall_f1) / len(overall_f1)),
        "overall_f1_per_ckpt": overall_f1,
        "class_wise_mean": {
            c: float(sum(r["class_wise"][c] for r in per_ckpt) / len(per_ckpt)) for c in classes
        },
        "average_class_f1_mean": float(
            sum(r["average_class_f1"] for r in per_ckpt) / len(per_ckpt)
        ),
    }
    return summary


def train(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating module <{cfg.modules._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.modules)

    log.info("Instantiating callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers ...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer ...")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log_hyperparameters(
        {"config": cfg, "model": model, "trainer": trainer, "callbacks": callbacks, "logger": logger}
    )

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if not trainer.is_global_zero:
        return

    # Reproduce Table-2 protocol: mean over the best-K checkpoints by val loss.
    ckpt_cb = next((c for c in callbacks if c.__class__.__name__ == "ModelCheckpoint"), None)
    ckpt_paths = []
    if ckpt_cb is not None and getattr(ckpt_cb, "best_k_models", None):
        ckpt_paths = list(ckpt_cb.best_k_models.keys())
    if not ckpt_paths and ckpt_cb is not None and ckpt_cb.best_model_path:
        ckpt_paths = [ckpt_cb.best_model_path]
    if not ckpt_paths:
        log.warning("No checkpoints found; skipping segment-F1 evaluation.")
        return

    log.info(f"Evaluating segment-wise F1 over {len(ckpt_paths)} checkpoints ...")
    test_summary = _evaluate_checkpoints(cfg, model, datamodule, trainer, ckpt_paths, "test")

    out = {
        "experiment_name": cfg.get("experiment_name"),
        "seed": cfg.get("seed"),
        "test": test_summary,
    }
    out_path = os.path.join(cfg.paths.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Test overall segment F1 (mean over {test_summary['n_checkpoints']} ckpts): "
             f"{test_summary['overall_f1_mean']:.2f}")
    log.info(f"Wrote {out_path}")


@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    train(cfg)


if __name__ == "__main__":
    main()
