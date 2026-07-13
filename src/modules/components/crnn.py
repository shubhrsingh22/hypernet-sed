"""CRNN and HCRNN architectures for polyphonic SED.

Reconstructed from Singh, Phan & Benetos, "Hypernetworks for Sound Event
Detection: A Proof-of-Concept" (Section 2.2):

* 4 CNN blocks: 3x3 conv -> BatchNorm -> ReLU -> 2x2 pooling, with channel
  counts (64, 128, 256, 512).
* The CNN output is collapsed along frequency to a 512-dimensional per-frame
  vector, which is fed to the recurrent block.
* Recurrent block is either a standard LSTM (the CRNN baseline; unidirectional
  hidden 256 or bidirectional hidden 512) or a dynamic HyperLSTM (the HCRNN;
  unidirectional main hidden 256, hyper hidden 64/128, embedding size 8).
* A time-distributed fully connected layer with sigmoid activation produces the
  per-frame, per-class probabilities. Logits are returned here; the sigmoid is
  folded into ``BCEWithLogitsLoss`` for numerical stability.

Reconstruction note: the paper states "pooling operation with kernel size of
2x2" and "the input to the LSTM unit is a 512-dimensional tensor". To reproduce
the 512-d recurrent input while preserving the per-frame time resolution needed
for strong-label SED, pooling is applied along the frequency axis (96 -> 6) with
the time axis preserved by default, followed by an adaptive average pool that
collapses the remaining frequency bins to 1 (512 channels x 1 = 512). Both the
time and frequency pooling factors are configurable.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from src.modules.components.hyperlstm import HyperLSTM


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_pool: int, freq_pool: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(time_pool, freq_pool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class CRNN(nn.Module):
    """Convolutional recurrent network for polyphonic SED.

    Args:
        num_classes: number of sound event classes (10 for URBAN-SED).
        n_mels: number of input mel bands (96 in the paper).
        cnn_channels: output channels of the 4 CNN blocks.
        time_pool: per-block time pooling factors (1 preserves frame rate).
        freq_pool: per-block frequency pooling factors.
        rnn_type: ``"lstm"`` for the CRNN baseline, ``"hyperlstm"`` for the HCRNN.
        rnn_hidden: main recurrent hidden size (256).
        bidirectional: only used for ``rnn_type == "lstm"``.
        hyper_hidden: hypernetwork hidden size (64 or 128); HCRNN only.
        hyper_n_z: hyper embedding size (8); HCRNN only.
        dropout: dropout applied to the recurrent output before the classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        n_mels: int = 96,
        cnn_channels: Sequence[int] = (64, 128, 256, 512),
        time_pool: Sequence[int] = (1, 1, 1, 1),
        freq_pool: Sequence[int] = (2, 2, 2, 2),
        rnn_type: str = "lstm",
        rnn_hidden: int = 256,
        bidirectional: bool = False,
        hyper_hidden: int = 128,
        hyper_n_z: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(cnn_channels) == len(time_pool) == len(freq_pool)
        self.rnn_type = rnn_type

        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch, tp, fp in zip(cnn_channels, time_pool, freq_pool):
            blocks.append(ConvBlock(in_ch, out_ch, tp, fp))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        # Collapse the remaining frequency bins to 1 -> per-frame vector of size cnn_channels[-1].
        self.freq_pool_final = nn.AdaptiveAvgPool2d((None, 1))
        rnn_input = cnn_channels[-1]

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            rnn_output = rnn_hidden * (2 if bidirectional else 1)
        elif rnn_type == "hyperlstm":
            self.bidirectional = bidirectional
            self.rnn = HyperLSTM(
                input_size=rnn_input,
                hidden_size=rnn_hidden,
                hyper_size=hyper_hidden,
                n_z=hyper_n_z,
            )
            if bidirectional:
                # Separate HyperLSTM over the time-reversed sequence.
                self.rnn_bwd = HyperLSTM(
                    input_size=rnn_input,
                    hidden_size=rnn_hidden,
                    hyper_size=hyper_hidden,
                    n_z=hyper_n_z,
                )
            rnn_output = rnn_hidden * (2 if bidirectional else 1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_output, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, n_mels) -> logits (batch, time_out, num_classes)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.cnn(x)  # (B, C, T', F')
        x = self.freq_pool_final(x)  # (B, C, T', 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, T', C)

        if self.rnn_type == "lstm":
            x, _ = self.rnn(x)
        else:
            fwd = self.rnn(x)
            if getattr(self, "bidirectional", False):
                bwd = self.rnn_bwd(x.flip(1)).flip(1)
                x = torch.cat([fwd, bwd], dim=-1)
            else:
                x = fwd

        x = self.dropout(x)
        return self.classifier(x)  # logits
