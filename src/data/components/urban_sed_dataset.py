"""Chunk-level dataset over precomputed URBAN-SED log-mel features.

The paper splits each spectrogram into fixed 2-second chunks with a 0.5-second
hop (Section 3.3). This dataset enumerates those chunks across all files and
returns z-score-normalised feature chunks with their frame-level targets.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class UrbanSedChunkDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        mean: float,
        std: float,
        chunk_frames: int,
        hop_frames: int,
    ):
        self.records = records
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0
        self.chunk_frames = chunk_frames
        self.hop_frames = hop_frames

        # Precompute a flat index of (record_idx, start_frame) chunks.
        self.index: List[Tuple[int, int]] = []
        for ri, rec in enumerate(records):
            n = rec["n_frames"]
            if n <= chunk_frames:
                self.index.append((ri, 0))
                continue
            starts = list(range(0, n - chunk_frames + 1, hop_frames))
            if starts[-1] != n - chunk_frames:
                starts.append(n - chunk_frames)
            for s in starts:
                self.index.append((ri, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ri, start = self.index[i]
        rec = self.records[ri]
        feat = np.load(rec["feat"], mmap_mode="r")
        label = np.load(rec["label"], mmap_mode="r")
        n = feat.shape[0]
        end = min(start + self.chunk_frames, n)

        x = np.asarray(feat[start:end], dtype=np.float32)
        y = np.asarray(label[start:end], dtype=np.float32)
        if x.shape[0] < self.chunk_frames:
            pad = self.chunk_frames - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            y = np.pad(y, ((0, pad), (0, 0)), mode="constant")

        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(y)
