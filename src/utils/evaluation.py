"""File-level SED evaluation.

Reassembles per-chunk model predictions into full-clip frame probabilities via
overlap-add averaging, converts them to event lists, and scores them against the
reference annotations with the segment-wise F1 (1-second segments).
"""

from typing import Dict, List

import numpy as np
import torch

from src.utils.sed_eval_utils import activity_to_events, segment_based_f1


@torch.no_grad()
def predict_file(
    net: torch.nn.Module,
    feat: np.ndarray,
    mean: float,
    std: float,
    chunk_frames: int,
    hop_frames: int,
    device: torch.device,
    max_batch: int = 128,
) -> np.ndarray:
    """Return full-clip frame probabilities (T, C) via overlap-add averaging."""
    n = feat.shape[0]
    std = std if std > 1e-8 else 1.0
    x = (feat.astype(np.float32) - mean) / std

    if n <= chunk_frames:
        starts = [0]
    else:
        starts = list(range(0, n - chunk_frames + 1, hop_frames))
        if starts[-1] != n - chunk_frames:
            starts.append(n - chunk_frames)

    chunks = []
    for s in starts:
        c = x[s : s + chunk_frames]
        if c.shape[0] < chunk_frames:
            c = np.pad(c, ((0, chunk_frames - c.shape[0]), (0, 0)), mode="constant")
        chunks.append(c)
    chunks_t = torch.from_numpy(np.stack(chunks)).to(device)

    probs_list = []
    for i in range(0, len(chunks_t), max_batch):
        logits = net(chunks_t[i : i + max_batch])
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)  # (n_chunks, chunk_frames, C)

    n_classes = probs.shape[-1]
    acc = np.zeros((n, n_classes), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)
    for k, s in enumerate(starts):
        end = min(s + chunk_frames, n)
        acc[s:end] += probs[k, : end - s]
        cnt[s:end] += 1.0
    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


@torch.no_grad()
def evaluate_split(
    net: torch.nn.Module,
    records: List[Dict],
    stats: Dict,
    chunk_frames: int,
    hop_frames: int,
    device: torch.device,
    threshold: float = 0.5,
    median_filter_frames: int = 0,
    time_resolution: float = 1.0,
) -> Dict:
    """Evaluate segment-wise F1 over a split (list of manifest records)."""
    net.eval()
    classes = stats["classes"]
    hop_sec = stats["hop_sec"]
    mean, std = stats["mean"], stats["std"]

    ref_all, est_all = [], []
    for rec in records:
        feat = np.load(rec["feat"])
        label = np.load(rec["label"]).astype(np.float32)
        probs = predict_file(net, feat, mean, std, chunk_frames, hop_frames, device)

        est = activity_to_events(probs, hop_sec, classes, threshold, median_filter_frames)
        ref = activity_to_events(label, hop_sec, classes, threshold=0.5)
        for e in est + ref:
            e["filename"] = rec["id"]
        est_all.append(est)
        ref_all.append(ref)

    return segment_based_f1(ref_all, est_all, classes, time_resolution)
