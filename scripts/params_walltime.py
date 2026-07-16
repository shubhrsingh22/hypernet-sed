"""Parameter counts and walltime benchmark for the URBAN-SED models (Table 2).

Measures, for each model on a single GPU in float32 (no JIT / torch.compile):
  - trainable parameter count
  - training step time: forward + BCE loss + backward + Adam step
  - inference time: forward only under torch.no_grad()

Input shape matches training: batch 64 chunks of 2 s (172 frames x 96 mels).

Usage:
    python scripts/params_walltime.py [--batch_size 64] [--device cuda:0] \
        [--out results/params_walltime.md]
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.components.crnn import CRNN  # noqa: E402

MODELS = {
    "crnn_uni": dict(rnn_type="lstm", bidirectional=False),
    "crnn_bi": dict(rnn_type="lstm", bidirectional=True, rnn_hidden=512),
    "hcrnn32": dict(rnn_type="hyperlstm", hyper_hidden=32),
    "hcrnn64": dict(rnn_type="hyperlstm", hyper_hidden=64),
    "hcrnn128": dict(rnn_type="hyperlstm", hyper_hidden=128),
    "hcrnn256": dict(rnn_type="hyperlstm", hyper_hidden=256),
}

# 2 s chunks at 22.05 kHz, hop 256 -> 172 frames of 96 log-mel bands
FRAMES, MELS, NUM_CLASSES = 172, 96, 10


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_loop(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def benchmark(name, kwargs, batch_size, device):
    torch.manual_seed(0)
    model = CRNN(num_classes=NUM_CLASSES, n_mels=MELS, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(batch_size, FRAMES, MELS, device=device)
    y = torch.randint(0, 2, (batch_size, FRAMES, NUM_CLASSES), device=device).float()
    opt = torch.optim.Adam(model.parameters(), lr=4e-5)
    crit = torch.nn.BCEWithLogitsLoss()

    model.train()

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    train_ms = time_loop(train_step, warmup=5, iters=20, device=device)

    model.eval()

    def infer_step():
        with torch.no_grad():
            model(x)

    infer_ms = time_loop(infer_step, warmup=10, iters=50, device=device)

    return dict(model=name, params=n_params,
                train_ms_per_step=round(train_ms, 2),
                infer_ms_per_batch=round(infer_ms, 2),
                infer_ms_per_clip=round(infer_ms / batch_size, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/params_walltime.md")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = []
    for name, kwargs in MODELS.items():
        rows.append(benchmark(name, kwargs, args.batch_size, device))
        print(rows[-1])

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    header = (
        f"# Parameter counts and walltime (URBAN-SED models)\n\n"
        f"GPU: {gpu} | PyTorch {torch.__version__} (CUDA {torch.version.cuda}) | "
        f"float32, no JIT/torch.compile | batch {args.batch_size} x 2 s chunks "
        f"({FRAMES} frames x {MELS} mels) | Adam, BCE loss.\n"
        f"Train step = forward+backward+optimizer; inference under no_grad. "
        f"Mean over 20 (train) / 50 (inference) iterations after warmup.\n\n"
    )
    table = "| Model | Params (M) | Train step (ms) | Inference (ms/batch) | Inference (ms/clip) |\n"
    table += "|---|---|---|---|---|\n"
    for r in rows:
        table += (f"| {r['model']} | {r['params']/1e6:.2f} | {r['train_ms_per_step']} "
                  f"| {r['infer_ms_per_batch']} | {r['infer_ms_per_clip']} |\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(header + table)
    with open(args.out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
