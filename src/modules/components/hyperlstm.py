"""Dynamic HyperLSTM (LayerNorm variant, faithful to hardmaru/supercell).

PyTorch port of the reference HyperLSTM implementation
(https://github.com/hardmaru/supercell), which is the canonical implementation
of the dynamic hypernetwork from Ha, Dai & Le, "HyperNetworks" (ICLR 2017), and
the one underlying Singh, Phan & Benetos, "Hypernetworks for Sound Event
Detection: A Proof-of-Concept".

Faithful details preserved from the reference:
* A small "hyper" LSTM runs alongside the "main" LSTM; its input is the
  concatenation of the main input ``x_t`` and previous main hidden ``h_{t-1}``.
* From the hyper hidden state, per-gate context vectors modulate the main cell:
  - ``hyper_norm``: weight scaling ``alpha`` applied to both the input (``W_xh``)
    and recurrent (``W_hh``) gate contributions. Init trick: the embedding ``z``
    starts at 1.0 (zero weights, bias 1.0) and ``alpha`` is initialised small
    (gamma 0.10 / embedding_size), matching recurrent batch-norm init.
  - ``hyper_bias``: additive bias ``beta`` (embedding init gaussian 0.01, output
    projection init 0).
* Layer normalisation in both the main and hyper cells (LN over each gate's
  pre-activation and over the cell state before the output tanh).
* Orthogonal init for recurrent weights; forget-bias 1.0; gate order (i, j, f, o).

The recurrence is sequential, but the input-only projections are precomputed for
the whole sequence and the four gates are handled in a single batched op; the
time loop is TorchScript-compiled to minimise kernel-launch overhead.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-3


@torch.jit.script
def _ln(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Layer norm over the last dim. Works for (B, H) and (B, 4, H)."""
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + 1e-3) * gamma + beta


@torch.jit.script
def _hyperlstm_loop(
    xh_seq: torch.Tensor,       # (B,T,4H)   main input projection (precomputed)
    xhyp_seq: torch.Tensor,     # (B,T,4Hy)  hyper input x-part (precomputed)
    h: torch.Tensor, c: torch.Tensor, h_hat: torch.Tensor, c_hat: torch.Tensor,
    # main cell
    W_hh: torch.Tensor, bias: torch.Tensor,
    ln_g: torch.Tensor, ln_b: torch.Tensor, lnc_g: torch.Tensor, lnc_b: torch.Tensor,
    # hyper cell
    W_xhyp_h: torch.Tensor, W_hh_hyper: torch.Tensor, bias_hyper: torch.Tensor,
    lnh_g: torch.Tensor, lnh_b: torch.Tensor, lnhc_g: torch.Tensor, lnhc_b: torch.Tensor,
    # hyper_norm (weight scaling) for input and recurrent contributions
    W_zw_x: torch.Tensor, b_zw_x: torch.Tensor, W_alpha_x: torch.Tensor,
    W_zw_h: torch.Tensor, b_zw_h: torch.Tensor, W_alpha_h: torch.Tensor,
    # hyper_bias
    W_zb: torch.Tensor, W_beta: torch.Tensor,
    hidden_size: int, hyper_size: int, n_z: int, forget_bias: float,
) -> torch.Tensor:
    B = xh_seq.shape[0]
    T = xh_seq.shape[1]
    outputs: List[torch.Tensor] = []
    for t in range(T):
        # --- hyper LSTM cell (LayerNorm LSTM) ---
        hpre = xhyp_seq[:, t] + F.linear(h, W_xhyp_h) + F.linear(h_hat, W_hh_hyper) + bias_hyper
        hpre = _ln(hpre.view(B, 4, hyper_size), lnh_g, lnh_b)
        hi = hpre[:, 0]; hj = hpre[:, 1]; hf = hpre[:, 2]; ho = hpre[:, 3]
        c_hat = c_hat * torch.sigmoid(hf + forget_bias) + torch.sigmoid(hi) * torch.tanh(hj)
        h_hat = torch.tanh(_ln(c_hat, lnhc_g, lnhc_b)) * torch.sigmoid(ho)

        # --- context vectors from hyper hidden state ---
        zw_x = (F.linear(h_hat, W_zw_x) + b_zw_x).view(B, 4, n_z)
        zw_h = (F.linear(h_hat, W_zw_h) + b_zw_h).view(B, 4, n_z)
        zb = F.linear(h_hat, W_zb).view(B, 4, n_z)
        alpha_x = torch.einsum("bgz,gzh->bgh", zw_x, W_alpha_x)
        alpha_h = torch.einsum("bgz,gzh->bgh", zw_h, W_alpha_h)
        beta = torch.einsum("bgz,gzh->bgh", zb, W_beta)

        # --- main LSTM cell with context-dependent weights/bias ---
        xh = xh_seq[:, t].view(B, 4, hidden_size)
        hh = F.linear(h, W_hh).view(B, 4, hidden_size)
        gates = alpha_x * xh + alpha_h * hh + bias.view(4, hidden_size) + beta
        gates = _ln(gates, ln_g, ln_b)
        gi = gates[:, 0]; gj = gates[:, 1]; gf = gates[:, 2]; go = gates[:, 3]
        c = c * torch.sigmoid(gf + forget_bias) + torch.sigmoid(gi) * torch.tanh(gj)
        h = torch.tanh(_ln(c, lnc_g, lnc_b)) * torch.sigmoid(go)
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def _ortho_lstm(out4: int, in_dim: int) -> torch.Tensor:
    """(in_dim, 4*hidden) with each hidden block orthogonally initialised."""
    hidden = out4 // 4
    w = torch.empty(in_dim, out4)
    for k in range(4):
        block = torch.empty(in_dim, hidden)
        nn.init.orthogonal_(block)
        w[:, k * hidden:(k + 1) * hidden] = block
    return w


class HyperLSTM(nn.Module):
    """Single-layer, unidirectional LayerNorm dynamic HyperLSTM over a sequence.

    Input ``(batch, time, input_size)`` -> output ``(batch, time, hidden_size)``.
    ``n_z`` is the hyper embedding size (8 in the paper). ``hyper_size`` is the
    hypernetwork hidden size (32/64/128/256).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        forget_bias: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size
        self.n_z = n_z
        self.forget_bias = forget_bias
        H, Hy, E = hidden_size, hyper_size, n_z
        init_gamma = 0.10

        # Main cell. W_xh stored transposed for F.linear (out, in).
        self.W_xh = nn.Parameter(_ortho_lstm(4 * H, input_size).t().contiguous())
        nn.init.xavier_uniform_(self.W_xh)  # input weights: uniform/glorot
        self.W_hh = nn.Parameter(_ortho_lstm(4 * H, H).t().contiguous())
        self.bias = nn.Parameter(torch.zeros(4 * H))
        self.ln_g = nn.Parameter(torch.ones(4, H)); self.ln_b = nn.Parameter(torch.zeros(4, H))
        self.lnc_g = nn.Parameter(torch.ones(H)); self.lnc_b = nn.Parameter(torch.zeros(H))

        # Hyper cell (LayerNorm LSTM); input is [x, h] of size input+H.
        self.W_xhyp_x = nn.Parameter(torch.empty(4 * Hy, input_size))
        nn.init.xavier_uniform_(self.W_xhyp_x)
        self.W_xhyp_h = nn.Parameter(torch.empty(4 * Hy, H))
        nn.init.xavier_uniform_(self.W_xhyp_h)
        self.W_hh_hyper = nn.Parameter(_ortho_lstm(4 * Hy, Hy).t().contiguous())
        self.bias_hyper = nn.Parameter(torch.zeros(4 * Hy))
        self.lnh_g = nn.Parameter(torch.ones(4, Hy)); self.lnh_b = nn.Parameter(torch.zeros(4, Hy))
        self.lnhc_g = nn.Parameter(torch.ones(Hy)); self.lnhc_b = nn.Parameter(torch.zeros(Hy))

        # hyper_norm: weight scaling for input (x) and recurrent (h) contributions.
        # z embedding starts at 1.0 (zero weights, bias 1.0); alpha init small.
        self.W_zw_x = nn.Parameter(torch.zeros(4 * E, Hy)); self.b_zw_x = nn.Parameter(torch.ones(4 * E))
        self.W_zw_h = nn.Parameter(torch.zeros(4 * E, Hy)); self.b_zw_h = nn.Parameter(torch.ones(4 * E))
        self.W_alpha_x = nn.Parameter(torch.full((4, E, H), init_gamma / E))
        self.W_alpha_h = nn.Parameter(torch.full((4, E, H), init_gamma / E))

        # hyper_bias: embedding gaussian(0.01), output projection init 0.
        self.W_zb = nn.Parameter(torch.randn(4 * E, Hy) * 0.01)
        self.W_beta = nn.Parameter(torch.zeros(4, E, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xh_seq = F.linear(x, self.W_xh)       # (B,T,4H)
        xhyp_seq = F.linear(x, self.W_xhyp_x)  # (B,T,4Hy)
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)
        h_hat = x.new_zeros(B, self.hyper_size)
        c_hat = x.new_zeros(B, self.hyper_size)
        return _hyperlstm_loop(
            xh_seq, xhyp_seq, h, c, h_hat, c_hat,
            self.W_hh, self.bias, self.ln_g, self.ln_b, self.lnc_g, self.lnc_b,
            self.W_xhyp_h, self.W_hh_hyper, self.bias_hyper,
            self.lnh_g, self.lnh_b, self.lnhc_g, self.lnhc_b,
            self.W_zw_x, self.b_zw_x, self.W_alpha_x,
            self.W_zw_h, self.b_zw_h, self.W_alpha_h,
            self.W_zb, self.W_beta,
            self.hidden_size, self.hyper_size, self.n_z, self.forget_bias,
        )
