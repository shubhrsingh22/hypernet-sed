"""Feature extraction for URBAN-SED.

Reproduces the feature pipeline in the paper (Section 3.3):

* 96-band log-mel spectrogram, window 1024 samples, hop 256 samples,
  sampling rate 22.05 kHz, frequency range 50-11025 Hz.
* Global z-score normalisation using the mean/std of the training set
  (applied on the fly at load time by the dataset).
* Strong (frame-level) targets built from the URBAN-SED annotations.

Outputs, under ``--out_dir``:
    features/<split>/<id>.npy   float32 (T, n_mels) unnormalised log-mel
    labels/<split>/<id>.npy     uint8   (T, n_classes) frame-level targets
    manifest.json               per-split list of {id, feat, label, n_frames}
    stats.json                  {mean, std, feature params, classes}

Usage:
    python -m src.preprocess --dataset_root /path/URBAN-SED_v2.0.0 \
        --out_dir /path/features [--limit N] [--num_workers 8]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.sed_eval_utils import URBAN_SED_CLASSES

SPLITS = ["train", "validate", "test"]

# Feature defaults (paper Section 3.3).
SR = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 96
FMIN = 50
FMAX = 11025


def parse_annotation(path: str) -> List[Tuple[float, float, str]]:
    """Parse a URBAN-SED annotation .txt (onset<TAB>offset<TAB>label)."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            try:
                onset = float(parts[0])
                offset = float(parts[1])
            except ValueError:
                continue  # header row
            label = parts[2].strip()
            events.append((onset, offset, label))
    return events


def events_to_frames(
    events: List[Tuple[float, float, str]],
    n_frames: int,
    hop_sec: float,
    class_to_idx: Dict[str, int],
) -> np.ndarray:
    """Build a (T, C) frame-level binary target from annotation events."""
    labels = np.zeros((n_frames, len(class_to_idx)), dtype=np.uint8)
    for onset, offset, label in events:
        if label not in class_to_idx:
            continue
        c = class_to_idx[label]
        start = int(np.floor(onset / hop_sec))
        end = int(np.ceil(offset / hop_sec))
        start = max(0, min(start, n_frames))
        end = max(0, min(end, n_frames))
        labels[start:end, c] = 1
    return labels


def compute_logmel(wav_path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=1.0).astype(np.float32)  # (n_mels, T)
    return logmel.T  # (T, n_mels)


def _process_one(args) -> Optional[Dict]:
    wav_path, ann_path, feat_path, label_path, split = args
    class_to_idx = {c: i for i, c in enumerate(URBAN_SED_CLASSES)}
    hop_sec = HOP / SR
    try:
        logmel = compute_logmel(wav_path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed {wav_path}: {e}")
        return None
    n_frames = logmel.shape[0]
    events = parse_annotation(ann_path) if os.path.exists(ann_path) else []
    labels = events_to_frames(events, n_frames, hop_sec, class_to_idx)

    np.save(feat_path, logmel)
    np.save(label_path, labels)

    rec = {
        "id": os.path.splitext(os.path.basename(wav_path))[0],
        "feat": feat_path,
        "label": label_path,
        "n_frames": int(n_frames),
    }
    if split == "train":
        rec["_sum"] = float(logmel.sum())
        rec["_sumsq"] = float((logmel.astype(np.float64) ** 2).sum())
        rec["_count"] = int(logmel.size)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Path to URBAN-SED_v2.0.0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--limit", type=int, default=0, help="Limit files/split (smoke test)")
    args = ap.parse_args()

    audio_root = os.path.join(args.dataset_root, "audio")
    ann_root = os.path.join(args.dataset_root, "annotations")
    hop_sec = HOP / SR

    manifest: Dict[str, List[Dict]] = {}
    train_sum = train_sumsq = 0.0
    train_count = 0

    for split in SPLITS:
        audio_dir = os.path.join(audio_root, split)
        ann_dir = os.path.join(ann_root, split)
        feat_dir = os.path.join(args.out_dir, "features", split)
        label_dir = os.path.join(args.out_dir, "labels", split)
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        wavs = sorted(
            f for f in os.listdir(audio_dir)
            if f.endswith(".wav") and not f.startswith(".")
        )
        if args.limit:
            wavs = wavs[: args.limit]

        tasks = []
        for w in wavs:
            base = os.path.splitext(w)[0]
            tasks.append(
                (
                    os.path.join(audio_dir, w),
                    os.path.join(ann_dir, base + ".txt"),
                    os.path.join(feat_dir, base + ".npy"),
                    os.path.join(label_dir, base + ".npy"),
                    split,
                )
            )

        records: List[Dict] = []
        print(f"[{split}] processing {len(tasks)} files with {args.num_workers} workers ...")
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures)):
                rec = fut.result()
                if rec is None:
                    continue
                if split == "train":
                    train_sum += rec.pop("_sum")
                    train_sumsq += rec.pop("_sumsq")
                    train_count += rec.pop("_count")
                records.append(rec)
                if (i + 1) % 500 == 0:
                    print(f"  [{split}] {i + 1}/{len(tasks)}")
        records.sort(key=lambda r: r["id"])
        manifest[split] = records
        print(f"[{split}] done: {len(records)} files")

    mean = train_sum / max(1, train_count)
    var = train_sumsq / max(1, train_count) - mean**2
    std = float(np.sqrt(max(var, 1e-12)))

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    stats = {
        "mean": float(mean),
        "std": float(std),
        "sr": SR, "n_fft": N_FFT, "hop": HOP, "win": WIN,
        "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
        "hop_sec": hop_sec,
        "classes": URBAN_SED_CLASSES,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"train mean={mean:.4f} std={std:.4f}")
    print(f"wrote manifest.json and stats.json to {args.out_dir}")


if __name__ == "__main__":
    main()
