"""LightningDataModule for URBAN-SED.

Reads the precomputed feature manifest/stats produced by ``src.preprocess`` and
serves 2-second / 0.5-second-hop chunks for training and validation. The test
split is scored file-by-file by the evaluator (see ``src.utils.evaluation``),
so it is exposed here as raw records rather than a chunk dataloader.
"""

import json
import os
from typing import Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.urban_sed_dataset import UrbanSedChunkDataset


class UrbanSedDataModule(LightningDataModule):
    def __init__(
        self,
        feature_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        chunk_sec: float = 2.0,
        chunk_hop_sec: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        with open(os.path.join(feature_dir, "manifest.json")) as f:
            self.manifest: Dict[str, List[Dict]] = json.load(f)
        with open(os.path.join(feature_dir, "stats.json")) as f:
            self.stats = json.load(f)

        hop_sec = self.stats["hop_sec"]
        self.chunk_frames = int(round(chunk_sec / hop_sec))
        self.chunk_hop_frames = int(round(chunk_hop_sec / hop_sec))
        self.classes = self.stats["classes"]
        self.hop_sec = hop_sec

        self.train_dataset: Optional[UrbanSedChunkDataset] = None
        self.val_dataset: Optional[UrbanSedChunkDataset] = None

    def _make(self, split: str) -> UrbanSedChunkDataset:
        return UrbanSedChunkDataset(
            records=self.manifest[split],
            mean=self.stats["mean"],
            std=self.stats["std"],
            chunk_frames=self.chunk_frames,
            hop_frames=self.chunk_hop_frames,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None) and self.train_dataset is None:
            self.train_dataset = self._make("train")
            self.val_dataset = self._make("validate")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            shuffle=False,
        )
