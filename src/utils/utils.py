"""Instantiation helpers for callbacks/loggers and hyperparameter logging."""

import logging
from typing import Dict, List

import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if not callbacks_cfg:
        log.warning("No callbacks configured.")
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("callbacks_cfg must be a DictConfig")
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No loggers configured.")
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("logger_cfg must be a DictConfig")
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


@rank_zero_only
def log_hyperparameters(object_dict: Dict) -> None:
    config = OmegaConf.to_container(object_dict["config"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]
    if not trainer.logger:
        return
    hparams = {
        "model": config.get("modules"),
        "data": config.get("data"),
        "trainer": config.get("trainer"),
        "callbacks": config.get("callbacks"),
        "seed": config.get("seed"),
        "experiment_name": config.get("experiment_name"),
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    for lg in trainer.loggers:
        lg.log_hyperparams(hparams)
