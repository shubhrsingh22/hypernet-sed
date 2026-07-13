"""LightningModule for polyphonic SED.

Trains a CRNN/HCRNN with frame-level binary cross-entropy over the 10 URBAN-SED
classes. Optimiser and schedule follow the paper (Section 3.4): Adam with a
learning rate of 4e-5, batch size 64, up to 250 epochs, early stopping with a
patience of 10 (early stopping / checkpointing are configured via callbacks).
"""

from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric


class SEDModule(LightningModule):
    def __init__(self, net: nn.Module, optimizer, scheduler=None, compile: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def _step(self, batch):
        x, y = batch
        logits = self.forward(x)
        return self.criterion(logits, y)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
            }
        return {"optimizer": optimizer}
