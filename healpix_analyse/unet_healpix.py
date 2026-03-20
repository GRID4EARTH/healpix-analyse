"""
unet_healpix.py
===============
Gauge-equivariant spherical U-Net for HEALPix maps.

Architecture
------------
Encoder  : HealPixDoubleConv → [HealPixDown → HealPixDoubleConv] × L
Bottleneck: deepest HealPixDoubleConv
Decoder  : [HealPixUp → concat(skip) → HealPixDoubleConv] × L
Head     : Conv1d(G*f_0, T_out, 1) → residual cumulative forecasts

Building blocks
---------------
  HealPixDoubleConv  — two HealPixConv layers + GroupNorm + ReLU each
  HealPixUp          — nearest-neighbour ×2 upsampling (NESTED ordering)
  HealPixDown        — Gaussian-smooth or max-pool ×2 downsampling
  HealPixConv        — gauge-equivariant spherical convolution

All geometry is precomputed once at construction time.
The forward pass is fully differentiable (autograd-compatible).

Dependencies
------------
    pip install torch healpix-geo numpy
    # Plus the companion modules:
    #   healpix_analyse.down    → HealPixDown
    #   healpix_analyse.convol  → HealPixConv
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown

ArrayLike = Union[np.ndarray, torch.Tensor]


# ===========================================================================
# HealPixUp — nearest-neighbour ×2 upsampling
# ===========================================================================

class HealPixUp(nn.Module):
    """
    HEALPix ×2 upsampling via nearest-neighbour interpolation (NESTED ordering).

    In the NESTED pixel scheme, the 4 children of coarse pixel p are always
    indexed as {4p, 4p+1, 4p+2, 4p+3}.  Upsampling therefore reduces to a
    single ``repeat_interleave(4, dim=-1)`` call — no lookup tables needed.

    Parameters
    ----------
    nside_out : int
        Output (fine) HEALPix resolution.  The input resolution is nside_out//2.
    device : torch.device or None
        Target device for the operation.
    """

    def __init__(self, nside_out: int, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device      = torch.device(device)
        self.nside_out   = int(nside_out)
        self.nside_in    = self.nside_out // 2
        # Full-sphere canonical output pixel ids
        self._cell_ids_out = np.arange(12 * self.nside_out ** 2, dtype=np.int64)

    @property
    def cell_ids_out(self) -> np.ndarray:
        """NESTED pixel indices at the output (fine) resolution."""
        return self._cell_ids_out

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, C, N_coarse]

        Returns
        -------
        y : torch.Tensor, shape [B, C, 4·N_coarse]
        cell_ids_out : np.ndarray
        """
        x = x.to(self.device)
        # repeat_interleave(4, dim=-1):
        #   pixel p is broadcast to positions 4p, 4p+1, 4p+2, 4p+3 — exactly
        #   the NESTED children ordering.
        return x.repeat_interleave(4, dim=-1), self._cell_ids_out


# ===========================================================================
# HealPixDoubleConv — two spherical conv layers with normalisation
# ===========================================================================

class HealPixDoubleConv(nn.Module):
    """
    Two consecutive HealPixConv layers, each followed by GroupNorm and ReLU.

    Data flow
    ---------
    x  [B, C_in, K]
      → HealPixConv(C_in,  C_out)  → [B, G·C_out, K]  → GroupNorm → ReLU
      → HealPixConv(G·C_out, C_out) → [B, G·C_out, K]  → GroupNorm → ReLU

    After the block the number of channels is always G·C_out, regardless of
    what was fed in.

    Parameters
    ----------
    nside : int
        HEALPix resolution for both conv layers.
    in_channels : int
        Number of input channels C_in.
    out_channels : int
        Number of base output channels C_out per gauge.
        The effective output channel count is n_gauges · C_out.
    kernel_sz : int, default 3
        Stencil side length (must be odd).  P = kernel_sz².
    n_gauges : int, default 1
        Number of gauge orientations G.
    gauge_type : str, default "phi"
        One of "phi", "cosmo", "projected_ref", "two_ref".
    singularity_lonlat : tuple or list of tuples or None
        Singularity placement for "projected_ref" or "two_ref" gauges.
        See HealPixConv documentation.
    ellipsoid : str, default "WGS84"
        Reference ellipsoid for coordinate conversions.
    device, dtype : torch device / dtype.
    """

    def __init__(
        self,
        nside:              int,
        in_channels:        int,
        out_channels:       int,
        kernel_sz:          int                            = 3,
        n_gauges:           int                            = 1,
        gauge_type:         str                            = "phi",
        singularity_lonlat: Optional[Any]                  = None,
        ellipsoid:          str                            = "WGS84",
        device:             Optional[torch.device]         = None,
        dtype:              torch.dtype                    = torch.float32,
    ):
        super().__init__()

        mid_ch = n_gauges * out_channels   # channels between the two conv layers

        # --- first convolution ---
        self.conv1 = HealPixConv(
            nside, in_channels, out_channels,
            kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ellipsoid=ellipsoid, device=device, dtype=dtype,
        )
        g1 = _find_groups(mid_ch)
        self.norm1 = nn.GroupNorm(g1, mid_ch)

        # --- second convolution (mid_ch → out_channels with n_gauges) ---
        self.conv2 = HealPixConv(
            nside, mid_ch, out_channels,
            kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ellipsoid=ellipsoid, device=device, dtype=dtype,
        )
        g2 = _find_groups(mid_ch)
        self.norm2 = nn.GroupNorm(g2, mid_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, C_in, K]

        Returns
        -------
        torch.Tensor [B, G·C_out, K]
        """
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


def _find_groups(num_channels: int, max_groups: int = 8) -> int:
    """Return the largest divisor of num_channels that is ≤ max_groups."""
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return max(g, 1)


# ===========================================================================
# HealPixUNet
# ===========================================================================

class HealPixUNet(nn.Module):
    """
    Gauge-equivariant spherical U-Net for HEALPix maps.

    Architecture overview
    ---------------------
    Encoder:
      enc_convs[0]  : HealPixDoubleConv(C_in,       f[0])  at nside_levels[0]
      down_ops[0]   : HealPixDown(nside_levels[0])
      enc_convs[1]  : HealPixDoubleConv(G·f[0],     f[1])  at nside_levels[1]
      ...
      enc_convs[L]  : HealPixDoubleConv(G·f[L-1],   f[L])  at nside_levels[L]  ← bottleneck

    Decoder (reverse):
      up_ops[0]     : HealPixUp(nside_levels[L-1])
      concat skip   : channels = G·f[L] + G·f[L-1]
      dec_convs[0]  : HealPixDoubleConv(G·(f[L]+f[L-1]),  f[L-1])
      ...
      up_ops[L-1]   : HealPixUp(nside_levels[0])
      concat skip   : channels = G·f[1] + G·f[0]
      dec_convs[L-1]: HealPixDoubleConv(G·(f[1]+f[0]),    f[0])

    Head:
      out_conv      : Conv1d(G·f[0], T_out, 1)  — one channel per forecast horizon

    Residual cumulative forecasting
    --------------------------------
    The network predicts T_out *residuals* (anomaly increments) rather than
    absolute SST values.  Starting from the last observed SST, the forecast at
    horizon h is:

        SST_hat[h] = SST_last + Σ_{k=0}^{h} Δ[k]

    where Δ[k] = gate_scale · clamp(residuals[k], ±residual_clip).

    Parameters
    ----------
    nside : int
        Base HEALPix resolution (power of 2).
    in_channels : int
        Total input channels = time_steps · vars_per_t + 1 (the +1 is DEM/mask).
    out_channels : int
        Number of forecast horizons T_out.
    feature_channels : list of int
        Channel count per U-Net level.  len(feature_channels) determines the
        depth.  Example: [64, 128, 256] → 3 levels.
    vars_per_t : int, default 7
        Number of physical variables per time step.  SST is assumed to be the
        last variable (index vars_per_t-1).
    time_steps : int, default 12
        Number of input time steps.
    down_mode : {"smooth", "maxpool"}, default "smooth"
        Downsampling strategy passed to HealPixDown.
    weight_norm : {"l1", "l2", "none"}, default "l1"
        Weight normalisation for the down-sampling Gaussian kernel.
    residual_clip : float, default 4.0
        Hard clamp applied to the residual forecasts before accumulation.
    gate_scale : float, default 0.5
        Multiplicative scaling applied after clamping.
    tau : float, default 4.0
        Decay constant for the temporal loss weighting.
    kernel_sz : int, default 3
        Spherical stencil size (odd integer ≥ 1).
    n_gauges : int, default 1
        Number of gauge orientations for HealPixConv.
    gauge_type : str, default "phi"
        Gauge convention.  One of "phi", "cosmo", "projected_ref", "two_ref".
    singularity_lonlat : tuple / list of tuples / None
        Singularity placement for "projected_ref"/"two_ref" gauges.
    ellipsoid : str, default "WGS84"
        Reference ellipsoid for healpix_geo coordinate conversions.
    device : torch.device or None
        Target device.  Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point precision.
    """

    def __init__(
        self,
        nside:              int,
        in_channels:        int,
        out_channels:       int,
        feature_channels:   List[int],
        *,
        vars_per_t:         int           = 7,
        time_steps:         int           = 12,
        down_mode:          str           = "smooth",
        weight_norm:        str           = "l1",
        residual_clip:      float         = 4.0,
        gate_scale:         float         = 0.5,
        tau:                float         = 4.0,
        kernel_sz:          int           = 3,
        n_gauges:           int           = 1,
        gauge_type:         str           = "phi",
        singularity_lonlat: Optional[Any] = None,
        ellipsoid:          str           = "WGS84",
        device:             Optional[Union[str, torch.device]] = None,
        dtype:              torch.dtype   = torch.float32,
    ):
        super().__init__()

        # ---- Device & dtype -----------------------------------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device  = torch.device(device)
        self.dtype   = dtype

        # ---- Validate depth ------------------------------------------------
        if len(feature_channels) < 1:
            raise ValueError("feature_channels must have at least one level.")

        # ---- Store hyperparameters -----------------------------------------
        self.nside              = int(nside)
        self.in_channels        = int(in_channels)
        self.out_channels       = int(out_channels)
        self.feature_channels   = list(feature_channels)
        self.vars_per_t         = int(vars_per_t)
        self.time_steps         = int(time_steps)
        self.down_mode          = str(down_mode)
        self.weight_norm        = str(weight_norm)
        self.residual_clip      = float(residual_clip)
        self.gate_scale         = float(gate_scale)
        self.tau                = float(tau)
        self.kernel_sz          = int(kernel_sz)
        self.n_gauges           = int(n_gauges)
        self.gauge_type         = str(gauge_type)
        self.singularity_lonlat = singularity_lonlat
        self.ellipsoid          = str(ellipsoid)

        # SST is the last physical variable: index vars_per_t - 1
        self.sst_index = self.vars_per_t - 1

        # Expected total input channels
        expected_ci = self.time_steps * self.vars_per_t + 1
        if self.in_channels != expected_ci:
            import warnings
            warnings.warn(
                f"in_channels={self.in_channels} ≠ time_steps*vars_per_t+1={expected_ci}. "
                "Make sure your input encoding matches the expected layout.",
                UserWarning, stacklevel=2,
            )

        # ---- Derive nside at each U-Net level ------------------------------
        #   nside_levels[0] = base nside (finest)
        #   nside_levels[L] = nside / 2^L (coarsest / bottleneck)
        self.nside_levels: List[int] = [self.nside]
        tmp = self.nside
        for _ in range(len(self.feature_channels) - 1):
            tmp //= 2
            if tmp < 1:
                raise ValueError(
                    "Too many levels in feature_channels for the given nside."
                )
            self.nside_levels.append(tmp)

        L = len(self.feature_channels) - 1   # bottleneck level index

        # ---- Helper for HealPixDoubleConv kwargs ---------------------------
        def _dconv_kwargs(lvl_nside, in_ch, out_ch):
            return dict(
                nside=lvl_nside,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_sz=self.kernel_sz,
                n_gauges=self.n_gauges,
                gauge_type=self.gauge_type,
                singularity_lonlat=self.singularity_lonlat,
                ellipsoid=self.ellipsoid,
                device=self.device,
                dtype=self.dtype,
            )

        # ---- Encoder -------------------------------------------------------
        self.enc_convs = nn.ModuleList()
        self.down_ops  = nn.ModuleList()

        # Level 0: in_channels → feature_channels[0]
        self.enc_convs.append(
            HealPixDoubleConv(**_dconv_kwargs(
                self.nside_levels[0], self.in_channels, self.feature_channels[0]
            ))
        )

        # Levels 1 .. L: downsample then double-conv
        for i in range(L):
            # Down from level i  →  level i+1
            self.down_ops.append(
                HealPixDown(
                    nside_in=self.nside_levels[i],
                    mode=self.down_mode,
                    weight_norm=self.weight_norm,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            # Double-conv at level i+1
            # Input: G·f[i] channels (output of previous enc_conv)
            in_ch = self.n_gauges * self.feature_channels[i]
            self.enc_convs.append(
                HealPixDoubleConv(**_dconv_kwargs(
                    self.nside_levels[i + 1], in_ch, self.feature_channels[i + 1]
                ))
            )

        # ---- Decoder -------------------------------------------------------
        self.up_ops   = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        # up_ops[j] upsamples from nside_levels[L-j] to nside_levels[L-j-1]
        # dec_convs[j] processes the concatenated tensor at nside_levels[L-j-1]
        for j in range(L):
            enc_lvl = L - 1 - j          # encoder level for the skip connection
            fine_nside = self.nside_levels[enc_lvl]

            self.up_ops.append(
                HealPixUp(nside_out=fine_nside, device=self.device)
            )

            # Channels entering the decoder double-conv:
            #   from upsampling   : G · f[enc_lvl + 1]
            #   from skip         : G · f[enc_lvl]
            in_ch  = self.n_gauges * (
                self.feature_channels[enc_lvl + 1] + self.feature_channels[enc_lvl]
            )
            out_ch = self.feature_channels[enc_lvl]
            self.dec_convs.append(
                HealPixDoubleConv(**_dconv_kwargs(fine_nside, in_ch, out_ch))
            )

        # ---- Output head ---------------------------------------------------
        # x_dec has G · f[0] channels after the last decoder conv
        final_ch = self.n_gauges * self.feature_channels[0]
        self.out_conv = nn.Conv1d(final_ch, self.out_channels, kernel_size=1)

        # Move everything to the target device
        self.to(self.device)

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _apply_down(
        down_op: HealPixDown, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply a HealPixDown operator to a [B, C, K] tensor.

        HealPixDown.forward expects [B, N] (2-D), so we reshape to [B·C, K],
        apply the operator, then restore to [B, C, K_out].  This is valid
        because the sparse matrix M is applied independently to each row.
        """
        B, C, K = feat.shape
        y, cell_ids = down_op(feat.reshape(B * C, K))
        K_out = y.shape[-1]
        return y.reshape(B, C, K_out), cell_ids

    # =========================================================================
    # Loss functions
    # =========================================================================

    @staticmethod
    def _corrcoef_1d(
        a: torch.Tensor,
        b: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Pearson correlation coefficient along the given dimension.

        Parameters
        ----------
        a, b : torch.Tensor — same shape
        dim  : int — dimension along which to compute correlation

        Returns
        -------
        torch.Tensor — same shape as a with `dim` reduced
        """
        a0  = a - a.mean(dim=dim, keepdim=True)
        b0  = b - b.mean(dim=dim, keepdim=True)
        num = (a0 * b0).sum(dim=dim)
        den = torch.sqrt((a0 ** 2).sum(dim=dim) * (b0 ** 2).sum(dim=dim) + eps)
        return num / den

    def _temporal_weights(
        self, T_out: int, device: torch.device
    ) -> torch.Tensor:
        """
        Exponentially decaying weights over forecast horizons: w[k] ∝ exp(−k/τ).

        Returns
        -------
        torch.Tensor [T_out] — normalised so that w.sum() = 1
        """
        k = torch.arange(T_out, device=device, dtype=torch.float32)
        w = torch.exp(-k / self.tau)
        return w / (w.sum() + 1e-12)

    def loss_mse_temporal(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temporally weighted mean-squared error.

        Each forecast horizon h contributes exp(−h/τ) / Σ exp(−k/τ)
        to the total loss.  Near-term horizons are penalised more heavily.

        Parameters
        ----------
        y_pred, y_true : torch.Tensor [B, T_out, K]

        Returns
        -------
        torch.Tensor — scalar loss
        """
        assert y_pred.shape == y_true.shape, (
            f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}"
        )
        B, T_out, K = y_pred.shape
        err  = (y_pred - y_true) ** 2                               # [B, T_out, K]
        w    = self._temporal_weights(T_out, y_pred.device)         # [T_out]
        loss = (w.view(1, T_out, 1) * err).mean()
        return loss

    def loss_corr_spatial(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        eps:    float = 1e-8,
    ) -> torch.Tensor:
        """
        Temporally weighted spatial correlation loss:  1 − mean_corr(space).

        Spatial Pearson correlation is computed over K pixels per (batch, horizon)
        pair, then averaged over batches with exponential horizon weights.

        Parameters
        ----------
        y_pred, y_true : torch.Tensor [B, T_out, K]

        Returns
        -------
        torch.Tensor — scalar in [0, 2]  (0 = perfect correlation)
        """
        assert y_pred.shape == y_true.shape
        B, T_out, K = y_pred.shape

        corr_bt = self._corrcoef_1d(y_pred, y_true, dim=-1, eps=eps)  # [B, T_out]
        w       = self._temporal_weights(T_out, y_pred.device)         # [T_out]
        return 1.0 - (corr_bt * w.unsqueeze(0)).sum(dim=1).mean()

    def loss_corr_plus_zmse(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        lam:    float = 0.05,
        eps:    float = 1e-8,
    ) -> torch.Tensor:
        """
        Combined loss:  (1 − spatial_corr) + λ · MSE_on_standardised_anomalies.

        The z-score MSE term helps the network preserve spatial variability
        without dominating the correlation objective.

        Parameters
        ----------
        y_pred, y_true : torch.Tensor [B, T_out, K]
        lam            : float — weight of the z-MSE term
        """
        lc = self.loss_corr_spatial(y_pred, y_true, eps=eps)

        mu_p  = y_pred.mean(dim=-1, keepdim=True)
        mu_t  = y_true.mean(dim=-1, keepdim=True)
        std_p = y_pred.std(dim=-1, keepdim=True)  + eps
        std_t = y_true.std(dim=-1, keepdim=True)  + eps
        zp    = (y_pred - mu_p) / std_p
        zt    = (y_true - mu_t) / std_t
        zmse  = (zp - zt).pow(2).mean()

        return lc + lam * zmse

    def temporal_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default training loss: temporally weighted MSE.

        Override this method or pass a custom loss to fit() to use a
        different objective (e.g. loss_corr_spatial, loss_corr_plus_zmse).

        Parameters
        ----------
        y_pred, y_true : torch.Tensor [B, T_out, K]

        Returns
        -------
        torch.Tensor — scalar
        """
        return self.loss_mse_temporal(y_pred, y_true)

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode → bottleneck → decode → residual cumulative SST forecast.

        Parameters
        ----------
        x : torch.Tensor [B, C_in, K]
            Input map.  The last channel (index -1) is the static field (DEM
            or land-sea mask); the first T·V channels are the time-stacked
            physical variables in the order [t0_v0, t0_v1, …, tT_v(V-1)].

        Returns
        -------
        y_pred : torch.Tensor [B, T_out, K]
            Cumulative SST forecasts at horizons 1 … T_out.
        """
        x = x.to(self.device)
        B, C, K = x.shape

        N_full = 12 * self.nside ** 2
        if K != N_full:
            raise ValueError(
                f"Expected full-sphere K={N_full} pixels at nside={self.nside}, got K={K}."
            )

        # ------------------------------------------------------------------
        # 1. Extract the last observed SST for residual forecasting
        # ------------------------------------------------------------------
        # phys: [B, T·V, K] — time-stacked physical variables
        # dem:  [B, 1,   K] — static field (last input channel)
        phys = x[:, :-1, :]    # [B, T·V, K]
        T, V = self.time_steps, self.vars_per_t

        if phys.shape[1] != T * V:
            raise ValueError(
                f"Expected phys channels T·V={T*V}, got {phys.shape[1]}."
            )

        # Last SST: last time step, SST variable (index sst_index)
        phys_4d  = phys.view(B, T, V, K)                 # [B, T, V, K]
        last_sst = phys_4d[:, -1, self.sst_index, :]      # [B, K]
        last_sst = last_sst.unsqueeze(1)                   # [B, 1, K]

        # ------------------------------------------------------------------
        # 2. Encoder
        # ------------------------------------------------------------------
        feat = x   # [B, C_in, K]
        enc_feats: List[torch.Tensor] = []

        # Level 0: at base nside
        feat = self.enc_convs[0](feat)     # [B, G·f[0], K]
        enc_feats.append(feat)

        # Levels 1 … L: downsample + double-conv
        for level in range(len(self.feature_channels) - 1):
            feat, _ = self._apply_down(self.down_ops[level], feat)
            # feat: [B, G·f[level], K_down]
            feat = self.enc_convs[level + 1](feat)
            # feat: [B, G·f[level+1], K_down]
            enc_feats.append(feat)

        # enc_feats[-1] is the bottleneck
        x_dec = enc_feats[-1]

        # ------------------------------------------------------------------
        # 3. Decoder
        # ------------------------------------------------------------------
        # up_ops[j] / dec_convs[j] process from coarser to finer levels
        for j in range(len(self.up_ops)):
            x_up, _  = self.up_ops[j](x_dec)        # [B, G·f[enc_lvl+1], K_fine]
            skip      = enc_feats[-(j + 2)]           # [B, G·f[enc_lvl],   K_fine]
            x_cat     = torch.cat([x_up, skip], dim=1)  # [B, G·(f[l+1]+f[l]), K_fine]
            x_dec     = self.dec_convs[j](x_cat)     # [B, G·f[enc_lvl], K_fine]

        # x_dec: [B, G·f[0], K]

        # ------------------------------------------------------------------
        # 4. Output head: residual prediction
        # ------------------------------------------------------------------
        residuals = self.out_conv(x_dec)              # [B, T_out, K]
        residuals = torch.clamp(
            residuals, min=-self.residual_clip, max=self.residual_clip
        ) * self.gate_scale

        # ------------------------------------------------------------------
        # 5. Cumulative forecasts
        #    SST_hat[h] = SST_last + Σ_{k=0}^{h-1} Δ[k]
        # ------------------------------------------------------------------
        T_out = residuals.shape[1]
        current = last_sst                        # [B, 1, K]
        outputs: List[torch.Tensor] = []
        for h in range(T_out):
            current = current + residuals[:, h:h+1, :]   # [B, 1, K]
            outputs.append(current)

        y_pred = torch.cat(outputs, dim=1)        # [B, T_out, K]
        return y_pred

    # =========================================================================
    # Training helpers
    # =========================================================================

    def fit(
        self,
        x_train:      ArrayLike,
        y_train:      ArrayLike,
        x_val:        Optional[ArrayLike] = None,
        y_val:        Optional[ArrayLike] = None,
        n_epoch:      int   = 100,
        batch_size:   int   = 16,
        lr:           float = 1e-3,
        weight_decay: float = 1e-6,
        optimizer:    str   = "adam",
        view_epoch:   int   = 10,
        loss_fn:      Optional[Any] = None,
    ) -> Dict[str, List]:
        """
        Mini-batch training loop with optional validation.

        Parameters
        ----------
        x_train, y_train : array-like [N, C_in, K] / [N, T_out, K]
            Training data.  Accepts numpy arrays or torch tensors.
        x_val, y_val : array-like or None
            Optional validation split (same shapes as training).
        n_epoch : int
            Number of epochs.
        batch_size : int
            Mini-batch size for Adam.  Full batch is used for LBFGS.
        lr : float
            Initial learning rate.
        weight_decay : float
            L2 regularisation weight for Adam.
        optimizer : {"adam", "lbfgs"}
            Optimiser to use.
        view_epoch : int
            Print loss every this many epochs (also always prints epoch 1 and n_epoch).
        loss_fn : callable or None
            Custom loss function with signature ``loss_fn(y_pred, y_true) → scalar``.
            Defaults to ``self.temporal_loss``.

        Returns
        -------
        history : dict with keys "train_loss" and "val_loss"
        """
        loss_fn = loss_fn or self.temporal_loss

        # Convert inputs to float tensors on the target device
        def _to_tensor(arr):
            if not torch.is_tensor(arr):
                arr = torch.from_numpy(np.asarray(arr))
            return arr.float().to(self.device)

        x_train = _to_tensor(x_train)
        y_train = _to_tensor(y_train)
        N = x_train.shape[0]

        has_val = (x_val is not None) and (y_val is not None)
        if has_val:
            x_val = _to_tensor(x_val)
            y_val = _to_tensor(y_val)

        # ---- Optimiser ----------------------------------------------------
        opt_name = optimizer.lower().strip()
        if opt_name == "adam":
            opt = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "lbfgs":
            opt = torch.optim.LBFGS(
                self.parameters(), lr=lr, max_iter=20, history_size=10
            )
        else:
            raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got '{optimizer}'.")

        history: Dict[str, List] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epoch + 1):
            self.train()

            # ---- Adam: mini-batch gradient descent --------------------------
            if opt_name == "adam":
                perm        = torch.randperm(N, device=self.device)
                epoch_loss  = 0.0
                n_batches   = 0

                for start in range(0, N, batch_size):
                    idx  = perm[start : start + batch_size]
                    xb, yb = x_train[idx], y_train[idx]

                    opt.zero_grad()
                    loss = loss_fn(self(xb), yb)
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    opt.step()

                    epoch_loss += loss.item()
                    n_batches  += 1

                train_loss = epoch_loss / max(n_batches, 1)

            # ---- LBFGS: full-batch ------------------------------------------
            else:
                def closure():
                    opt.zero_grad()
                    loss = loss_fn(self(x_train), y_train)
                    loss.backward()
                    return loss

                train_loss = opt.step(closure).item()

            # ---- Validation -------------------------------------------------
            val_loss = None
            if has_val:
                self.eval()
                with torch.no_grad():
                    Nv = x_val.shape[0]
                    if Nv <= batch_size or opt_name == "lbfgs":
                        val_loss = loss_fn(self(x_val), y_val).item()
                    else:
                        acc, nb = 0.0, 0
                        for start in range(0, Nv, batch_size):
                            xb = x_val[start : start + batch_size]
                            yb = y_val[start : start + batch_size]
                            acc += loss_fn(self(xb), yb).item()
                            nb  += 1
                        val_loss = acc / max(nb, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch == 1 or epoch % view_epoch == 0 or epoch == n_epoch:
                if val_loss is not None:
                    print(
                        f"[Epoch {epoch:4d}/{n_epoch}]  "
                        f"train={train_loss:.6f}  val={val_loss:.6f}"
                    )
                else:
                    print(f"[Epoch {epoch:4d}/{n_epoch}]  train={train_loss:.6f}")

        return history

    # =========================================================================
    # Inference helper
    # =========================================================================

    def predict(
        self, x: ArrayLike, batch_size: int = 16
    ) -> torch.Tensor:
        """
        Batched inference on arbitrary-size datasets.

        Parameters
        ----------
        x : array-like [N, C_in, K]
            Input maps (numpy array or torch tensor).
        batch_size : int
            Number of samples per forward pass.

        Returns
        -------
        torch.Tensor [N, T_out, K] on CPU.
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        x = x.float()

        self.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = x[start : start + batch_size].to(self.device)
                preds.append(self(xb).cpu())
        return torch.cat(preds, dim=0)

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def get_hparams(self) -> Dict[str, Any]:
        """Return the minimal set of hyperparameters needed to rebuild this model."""
        return {
            "nside":              int(self.nside),
            "in_channels":        self.in_channels,
            "out_channels":       self.out_channels,
            "feature_channels":   list(self.feature_channels),
            "vars_per_t":         self.vars_per_t,
            "time_steps":         self.time_steps,
            "down_mode":          self.down_mode,
            "weight_norm":        self.weight_norm,
            "residual_clip":      self.residual_clip,
            "gate_scale":         self.gate_scale,
            "tau":                self.tau,
            "kernel_sz":          self.kernel_sz,
            "n_gauges":           self.n_gauges,
            "gauge_type":         self.gauge_type,
            "singularity_lonlat": self.singularity_lonlat,
            "ellipsoid":          self.ellipsoid,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of (trainable) parameters."""
        fn = (lambda p: p.requires_grad) if trainable_only else (lambda p: True)
        return sum(p.numel() for p in self.parameters() if fn(p))

    def parameter_table(self) -> Dict[str, Any]:
        """Per-tensor parameter report suitable for run tracking."""
        rows, total, trainable = [], 0, 0
        for name, p in self.named_parameters():
            n         = p.numel()
            total    += n
            trainable += n if p.requires_grad else 0
            rows.append({
                "name":      name,
                "shape":     list(p.shape),
                "numel":     int(n),
                "trainable": bool(p.requires_grad),
            })
        return {"total": int(total), "trainable": int(trainable), "tensors": rows}

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert non-JSON types into JSON-serialisable equivalents."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [HealPixUNet._json_safe(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): HealPixUNet._json_safe(v) for k, v in obj.items()}
        return str(obj)

    def save_checkpoint(
        self,
        path:             str,
        *,
        history:          Optional[Dict[str, Any]]             = None,
        optimizer:        Optional[torch.optim.Optimizer]       = None,
        epoch:            Optional[int]                        = None,
        extra:            Optional[Dict[str, Any]]             = None,
        save_json_sidecar: bool                                = True,
    ) -> str:
        """
        Save a training checkpoint to disk.

        What is saved
        -------------
        - ``model_state``   — CPU state dict (portable across GPU configs)
        - ``hparams``       — all hyperparameters needed to rebuild the model
        - ``history``       — train / val loss arrays from fit()
        - ``optimizer_state`` — (optional) to resume training exactly
        - ``epoch``         — last completed epoch index
        - ``rng_state``     — PyTorch / NumPy / Python RNG states for reproducibility
        - ``param_report``  — parameter counts per tensor
        - ``versions``      — torch and numpy version strings

        A lightweight JSON sidecar is also written next to the .pt file for
        quick inspection without loading the full checkpoint.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "runs/epoch_10.pt").
        history : dict or None
            Training history dict, typically the return value of fit().
        optimizer : torch.optim.Optimizer or None
            If provided, its state dict is stored (enables exact resumption).
        epoch : int or None
            Last completed epoch number.
        extra : dict or None
            Any additional metadata (git hash, dataset id, notes, …).
        save_json_sidecar : bool
            Write a human-readable .json summary alongside the .pt file.

        Returns
        -------
        path : str
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        ckpt = {
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_class":     self.__class__.__name__,
            "hparams":         self.get_hparams(),
            "model_state":     {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "epoch":           epoch,
            "history":         history,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "param_report":    self.parameter_table(),
            "versions":        {"torch": torch.__version__, "numpy": np.__version__},
            "rng_state": {
                "torch_cpu":  torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy":      np.random.get_state(),
                "python":     random.getstate(),
            },
            "extra": extra or {},
        }

        torch.save(ckpt, path)

        if save_json_sidecar:
            meta = {k: ckpt[k] for k in (
                "timestamp", "model_class", "hparams", "epoch", "history", "versions", "extra"
            )}
            meta["param_report"] = {
                "total":     ckpt["param_report"]["total"],
                "trainable": ckpt["param_report"]["trainable"],
            }
            json_path = os.path.splitext(path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)

        return path

    @classmethod
    def from_checkpoint(
        cls,
        path:         str,
        *,
        device:       Optional[torch.device] = None,
        map_location: str                    = "cpu",
        strict:       bool                   = True,
    ) -> Tuple["HealPixUNet", Dict[str, Any]]:
        """
        Rebuild a model from a checkpoint created by save_checkpoint().

        Parameters
        ----------
        path : str
            Path to the .pt checkpoint file.
        device : torch.device or None
            Device on which to place the model after loading.
            Defaults to CUDA if available, else CPU.
        map_location : str
            Passed to torch.load (use "cpu" for portability, then move to GPU).
        strict : bool
            Passed to load_state_dict.

        Returns
        -------
        model : HealPixUNet
        ckpt  : dict — the full checkpoint dictionary (includes history, etc.)
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        h    = ckpt["hparams"].copy()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(
            nside             = int(h["nside"]),
            in_channels       = h["in_channels"],
            out_channels      = h["out_channels"],
            feature_channels  = h["feature_channels"],
            vars_per_t        = h.get("vars_per_t",        7),
            time_steps        = h.get("time_steps",        12),
            down_mode         = h.get("down_mode",         "smooth"),
            weight_norm       = h.get("weight_norm",       "l1"),
            residual_clip     = h.get("residual_clip",     4.0),
            gate_scale        = h.get("gate_scale",        0.5),
            tau               = h.get("tau",               4.0),
            kernel_sz         = h.get("kernel_sz",         3),
            n_gauges          = h.get("n_gauges",          1),
            gauge_type        = h.get("gauge_type",        "phi"),
            singularity_lonlat= h.get("singularity_lonlat", None),
            ellipsoid         = h.get("ellipsoid",         "WGS84"),
            device            = device,
        )

        model.load_state_dict(ckpt["model_state"], strict=strict)
        model.to(device)
        return model, ckpt

    # =========================================================================
    # Utilities
    # =========================================================================

    def extra_repr(self) -> str:
        return (
            f"nside={self.nside}, depth={len(self.feature_channels)}, "
            f"features={self.feature_channels}, "
            f"in={self.in_channels}, out={self.out_channels}, "
            f"G={self.n_gauges}, gauge={self.gauge_type!r}, "
            f"params={self.count_parameters():,}"
        )


# ===========================================================================
# Smoke-test
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    nside       = 16               # small for quick test
    npix        = 12 * nside ** 2  # 3 072 pixels
    T, V        = 4, 3             # 4 time steps, 3 variables → C_in = T*V+1 = 13
    T_out       = 6                # 6 forecast horizons

    model = HealPixUNet(
        nside            = nside,
        in_channels      = T * V + 1,
        out_channels     = T_out,
        feature_channels = [16, 32, 64],
        vars_per_t       = V,
        time_steps       = T,
        n_gauges         = 2,
        gauge_type       = "phi",
        kernel_sz        = 3,
        down_mode        = "smooth",
    )

    print(model)
    print(f"\nTrainable parameters : {model.count_parameters():,}")

    # --- quick forward check ---
    x = torch.randn(2, T * V + 1, npix)
    y = model(x)
    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(y.shape)}")   # expected [2, 6, 3072]

    # --- quick training check ---
    import numpy as np
    N = 8
    x_tr = np.random.randn(N, T * V + 1, npix).astype(np.float32)
    y_tr = np.random.randn(N, T_out,      npix).astype(np.float32)

    history = model.fit(
        x_tr, y_tr,
        n_epoch=3, batch_size=4, lr=1e-3, view_epoch=1,
    )
    print("\nTraining smoke-test passed.")
