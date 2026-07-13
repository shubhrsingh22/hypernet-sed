"""Standalone evaluation entrypoint.

Loads a trained checkpoint and reports segment-wise F1 (class-wise + overall) on
a chosen split. Example:

    python -m src.eval experiment=hcrnn128 ckpt_path=/path/to/ckpt.ckpt split=test
"""

import json
import logging

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from src.utils.evaluation import evaluate_split

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.get("ckpt_path"), "Provide ckpt_path=..."
    split = cfg.get("split", "test")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.modules)
    state = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = evaluate_split(
        model.net.to(device),
        datamodule.manifest[split],
        datamodule.stats,
        datamodule.chunk_frames,
        datamodule.chunk_hop_frames,
        device,
        threshold=cfg.get("eval_threshold", 0.5),
        median_filter_frames=cfg.get("median_filter_frames", 0),
    )
    print(json.dumps(res, indent=2))
    log.info(f"[{split}] overall segment F1 = {res['overall']['f_measure']:.2f}")


if __name__ == "__main__":
    main()
