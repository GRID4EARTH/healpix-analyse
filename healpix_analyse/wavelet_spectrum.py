"""Wavelet power spectrum on HEALPix, NaN- and mask-aware.

The spectrum is built from two existing operators:

- :class:`~healpix_analyse.convol.HealPixConv` applies a fixed band-pass
  wavelet (a complex Morlet in ``G`` orientations, or its angle average) at
  each level, with ``missing="local_mean"`` so that NaN values and pixels
  outside ``cell_ids`` are completed instead of propagated;
- :class:`~healpix_analyse.down.HealPixDown` takes the field one level down,
  carrying the pair (q, m) = (w (x - M), w) so that missing pixels are handled by a
  normalised convolution.  The default descent, ``mode="mean"`` made
  area-weighted, conserves the sums of (q, m) from level to level.

At every level ``j`` (fine to coarse, down to level 0), the band power is the
weighted mean of ``|y_j|^2`` over the valid pixels, with weights that account
for the part of the kernel that fell on missing pixels.  The mean of the
level-0 pixels, which is the mean of the valid input pixels, is returned as
the frequency-0 term, and their variance around it as the low-pass
residual.  Everything is PyTorch, differentiable with respect to the data,
and runs on CPU or GPU.  No SciPy.

Design and validation: ``docs/sht/wavelet_spectrum.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse.convol import HealPixConv, _DEFAULT_CACHE_DIR
from healpix_analyse.down import HealPixDown
from healpix_analyse._ellipsoid import canonicalize_ellipsoid

ArrayLike = Union[np.ndarray, torch.Tensor]
KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

_EARTH_RADIUS_KM = 6371.0
_ALPHA0 = math.sqrt(4.0 * math.pi / 12.0)       # pixel size at level 0 (rad)
_TRUNCATION_MAX = 1e-3                          # envelope energy allowed outside the stencil

# Default (k0 [rad/pixel], sigma [pixel]) per number of scales per octave.
# sigma * k0 ~ 1.9 Q keeps the overlap of neighbouring bands about constant.
_DEFAULTS_ANISO = {1: (2.4, 0.8), 2: (2.4, 1.6)}
_DEFAULTS_ISO = {1: (2.0, 1.0), 2: (2.4, 1.6)}


# ===========================================================================
# Wavelet profiles  (convention fn(rho_pix, phi_rad) of kernel_pyramid)
# ===========================================================================

def wavelet_morlet(sigma_pix: float = 0.8, k0: float = 2.4) -> KernelFn:
    """Complex Morlet profile ``fn(rho_pix, phi_rad)``.

    ``psi(rho, phi) = exp(-rho^2 / 2 sigma^2) (exp(i k0 rho cos phi) - exp(-k0^2 sigma^2 / 2))``

    ``rho`` in pixels, ``phi`` the angle from the first axis of the local
    stencil frame (the gauge direction), ``k0`` the carrier in radians per
    pixel.  The correction term makes the continuous profile zero-mean; the
    discrete taps are made zero-sum when they are sampled.
    """
    s, k = float(sigma_pix), float(k0)
    if s <= 0 or k <= 0:
        raise ValueError("sigma_pix and k0 must be positive.")
    corr = math.exp(-0.5 * (k * s) ** 2)

    def fn(rho, phi):
        rho = np.asarray(rho, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        env = np.exp(-0.5 * (rho / s) ** 2)
        return env * (np.exp(1j * k * rho * np.cos(phi)) - corr)

    fn.sigma_pix, fn.k0 = s, k
    return fn


def wavelet_morlet_isotropic(sigma_pix: float = 1.0, k0: float = 2.0,
                             n_angles: int = 64) -> KernelFn:
    """Angle average of :func:`wavelet_morlet`: a real, radial profile.

    ``psi(rho) = exp(-rho^2 / 2 sigma^2) (J0(k0 rho) - exp(-k0^2 sigma^2 / 2))``,
    computed as the mean of the oriented profile over ``n_angles`` angles,
    so that no Bessel function (and no SciPy) is needed.  It covers the same
    radial band as the oriented Morlet.
    """
    base = wavelet_morlet(sigma_pix, k0)
    angs = np.arange(int(n_angles)) * (2.0 * np.pi / int(n_angles))

    def fn(rho, phi=None):
        rho = np.asarray(rho, dtype=np.float64)
        vals = np.mean([base(rho, np.full_like(rho, a)) for a in angs], axis=0)
        return vals.real

    fn.sigma_pix, fn.k0 = base.sigma_pix, base.k0
    return fn


def _stencil_polar(kernel_sz: int) -> tuple[np.ndarray, np.ndarray]:
    """``(rho_pix, phi_rad)`` of the stencil points, in HealPixConv's order.

    Same grid as ``convol._local_kernel_grid`` (``meshgrid(grid, grid)``,
    flattened row by row), whose angular distances are ``rho * alpha``.
    """
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    return np.hypot(xx, yy).ravel().astype(np.float64), np.arctan2(yy, xx).ravel()


def _sample_taps(fn: KernelFn, kernel_sz: int, dilation: float = 1.0) -> np.ndarray:
    """Zero-sum, unit-L2 taps of ``fn(rho / dilation, phi)`` on the stencil."""
    rho, phi = _stencil_polar(kernel_sz)
    taps = np.asarray(fn(rho / dilation, phi))
    taps = taps - taps.mean()
    norm = np.sqrt(np.sum(np.abs(taps) ** 2))
    if norm == 0:
        raise ValueError("the wavelet taps are all zero on this stencil.")
    return taps / norm


def _envelope_truncation(kernel_sz: int, sigma_pix: float) -> float:
    """Fraction of the energy of exp(-rho^2/2 sigma^2) outside the stencil."""
    half = max(40, int(8 * sigma_pix) + kernel_sz)
    g = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(g, g)
    e = np.exp(-(xx ** 2 + yy ** 2) / sigma_pix ** 2)
    h = kernel_sz // 2
    inside = (np.abs(xx) <= h) & (np.abs(yy) <= h)
    return float(1.0 - e[inside].sum() / e.sum())


def _kernel_size_for(sigma_max: float) -> int:
    return 2 * int(math.ceil(2.2 * sigma_max)) + 1


def _corner_deg(kernel_sz: int, level: int) -> float:
    return math.degrees((kernel_sz // 2) * math.sqrt(2.0) * _ALPHA0 / 2 ** level)


# ===========================================================================
# Outputs
# ===========================================================================

@dataclass(frozen=True)
class WaveletSpectrum:
    """Output of :class:`HealPixWaveletSpectrum` (``...`` = leading dims of x).

    S = number of scales (levels x scales per octave), fine to coarse.
    G = number of orientations (absent for the isotropic class).
    """

    power: torch.Tensor            # [..., S, G] (isotropic: [..., S])
    power_total: torch.Tensor      # [..., S], summed over orientations
    s1: Optional[torch.Tensor]     # like power, mean |y|; None unless s1=True
    monopole: torch.Tensor         # [...], frequency 0: mean of the valid pixels
    lowpass_var: torch.Tensor      # [...], variance of the level-0 pixels around it
    n_eff: torch.Tensor            # [..., S], sum of the weights (smallest over orientations); 0 -> NaN
    levels: torch.Tensor           # [S], HEALPix level of each scale
    voices: torch.Tensor           # [S], voice q (0 .. Q-1) of each scale
    ell_nominal: torch.Tensor      # [S], k0 / (a_q * pixel size): nominal multipole
    scale_km: torch.Tensor         # [S], 2 pi R / (ell_nominal + 1/2)
    local_power: Optional[list]    # per level, [..., G*Q, N_j] maps of |y|^2


@dataclass(frozen=True)
class WaveletCrossSpectrum:
    """Output of ``cross``: co-spectrum and coherence per scale."""

    co: torch.Tensor               # [..., S, G] (isotropic: [..., S]), Re <y_x conj y_y>
    quad: Optional[torch.Tensor]   # [..., S, G], Im part (anisotropic only)
    co_total: torch.Tensor         # [..., S]
    coherence: torch.Tensor        # [..., S], co_total / sqrt(power_x * power_y)
    power_x: torch.Tensor          # [..., S], on the joint weights
    power_y: torch.Tensor          # [..., S], on the joint weights
    n_eff: torch.Tensor            # [..., S]
    levels: torch.Tensor
    voices: torch.Tensor


@dataclass(frozen=True)
class WaveletWindows:
    """Measured response of each scale to unit-power multipoles.

    For a statistically isotropic field with spectrum C_ell on the full
    sphere, ``E[power_total[s]] = sum_ell C_ell * response[s, ell]`` and
    ``E[lowpass_var] = sum_ell C_ell * lowpass[ell]``.
    """

    ell: torch.Tensor              # [lmax + 1]
    response: torch.Tensor         # [S, lmax + 1]
    lowpass: torch.Tensor          # [lmax + 1]
    ell_eff: torch.Tensor          # [S], centroid of response / (2 ell + 1)


@dataclass(frozen=True)
class LittlewoodPaley:
    """Band constants fitted so that the weighted band powers add up to the variance.

    ``(constants * power_total).sum(-1) + lowpass * lowpass_var`` estimates
    the variance of a full-sphere map around its mean, within ``ripple``
    (``(max - min) / mean`` of the fitted sum of windows over ell >= 1).
    """

    constants: torch.Tensor        # [S]
    lowpass: torch.Tensor          # []
    ripple: torch.Tensor           # []


# ===========================================================================
# Engine
# ===========================================================================

def _safe_div(num, den, eps=0.0):
    ok = den > eps
    return torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)),
                       torch.zeros_like(num)), ok


def _forked_rng(device: torch.device):
    """Context that restores the CPU RNG, and the RNG of ``device``, on exit."""
    if device.type == "cpu":
        return torch.random.fork_rng(devices=[])
    mod = getattr(torch, device.type, None)
    if mod is None or not hasattr(mod, "get_rng_state"):
        return torch.random.fork_rng(devices=[])
    idx = device.index
    if idx is None:
        idx = mod.current_device() if hasattr(mod, "current_device") else 0
    return torch.random.fork_rng(devices=[idx], device_type=device.type)


class _WaveletSpectrumBase(nn.Module):
    """Shared engine of the anisotropic and isotropic classes."""

    _isotropic = False

    def __init__(
        self,
        level: int,
        cell_ids: Optional[ArrayLike],
        n_scales: int,
        wavelet: Optional[KernelFn],
        scales_per_octave: int,
        kernel_sz: Optional[int],
        n_gauges: int,
        gauge_type: str,
        singularity_lonlat,
        ref_direction,
        down_mode: str,
        edge: str,
        coverage_weight: bool,
        min_coverage: float,
        min_valid: float,
        s1: bool,
        local_maps: bool,
        max_stencil_deg: float,
        ellipsoid: str,
        dtype: torch.dtype,
        device,
        cache_dir,
    ):
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        super().__init__()
        if isinstance(level, bool) or int(level) != level or int(level) < 1:
            raise ValueError("level must be an integer >= 1.")
        Q = int(scales_per_octave)
        if Q not in (1, 2):
            raise ValueError("scales_per_octave must be 1 or 2 (3 and more: not yet).")
        if edge not in ("mean_corrected", "complete", "none"):
            raise ValueError("edge must be 'mean_corrected', 'complete' or 'none'.")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1].")
        if not 0.0 < float(min_valid) <= 1.0:
            raise ValueError("min_valid must be in (0, 1].")
        if down_mode not in ("smooth", "mean"):
            # only these keep (q, m) a weighted mean of the valid pixels:
            # maxpool lets a missing child (q = 0) win, smooth_conservative
            # adds separate offsets to q and m that bias q/m near coasts
            raise ValueError("down_mode must be 'smooth' or 'mean'.")
        if (isinstance(n_scales, bool) or int(n_scales) != n_scales
                or not (int(n_scales) == -1 or int(n_scales) >= 1)):
            raise ValueError("n_scales must be -1 (all levels) or an integer >= 1.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.down_mode = down_mode
        self.level = int(level)
        self.Q = Q
        self.edge = edge
        self.coverage_weight = bool(coverage_weight)
        self.min_coverage = float(min_coverage)
        self.min_valid = float(min_valid)
        self.s1 = bool(s1)
        self.local_maps = bool(local_maps)
        self.G = 1 if self._isotropic else int(n_gauges)

        # ---- wavelet and kernel size ----
        defaults = _DEFAULTS_ISO if self._isotropic else _DEFAULTS_ANISO
        if wavelet is None:
            k0, sig = defaults[Q]
            wavelet = (wavelet_morlet_isotropic(sig, k0) if self._isotropic
                       else wavelet_morlet(sig, k0))
        self.wavelet = wavelet
        sig = getattr(wavelet, "sigma_pix", None)
        k0 = getattr(wavelet, "k0", None)
        dil = [2.0 ** (q / Q) for q in range(Q)]
        if kernel_sz is None:
            if sig is None:
                raise ValueError("kernel_sz is required for a custom wavelet "
                                 "without a sigma_pix attribute.")
            kernel_sz = _kernel_size_for(sig * dil[-1])
        kernel_sz = int(kernel_sz)
        if kernel_sz < 3 or kernel_sz % 2 == 0:
            raise ValueError("kernel_sz must be an odd integer >= 3.")
        if sig is not None:
            trunc = _envelope_truncation(kernel_sz, sig * dil[-1])
            if trunc > _TRUNCATION_MAX:
                raise ValueError(
                    f"kernel_sz={kernel_sz} cuts {100 * trunc:.2f} % of the envelope "
                    f"energy of the widest voice (sigma = {sig * dil[-1]:.2f} px); "
                    f"use kernel_sz >= {_kernel_size_for(sig * dil[-1])}."
                )
        self.kernel_sz = kernel_sz
        self.k0 = k0

        # fall-back 5x5 dyadic wavelet for the coarse levels where the wide
        # stencil would go past the antipode
        k0f, sigf = _DEFAULTS_ANISO[1]
        fallback = (wavelet_morlet_isotropic(sigf, k0f) if self._isotropic
                    else wavelet_morlet(sigf, k0f))

        J = self.level
        j_min = 0 if int(n_scales) == -1 else max(0, J - int(n_scales) + 1)
        self.band_levels = list(range(J, j_min - 1, -1))

        convs, downs, area, taps_l, band_q, wide = [], [], [], [], [], []
        ids = None if cell_ids is None else np.asarray(cell_ids, dtype=np.int64).ravel()
        self.partial = ids is not None
        self.n_cells = []
        for j in range(J, -1, -1):
            n_j = 12 * 4 ** j if ids is None else len(ids)
            self.n_cells.append(n_j)
            if j in self.band_levels:
                use_wide = _corner_deg(kernel_sz, j) <= float(max_stencil_deg)
                wide.append(use_wide)
                ksz = kernel_sz if use_wide else 5
                if use_wide:
                    taps = np.stack([_sample_taps(wavelet, ksz, d) for d in dil])
                    env_sig = (sig if sig is not None else 1.0)
                else:
                    taps = _sample_taps(fallback, ksz)[None]
                    env_sig = fallback.sigma_pix
                rho, _ = _stencil_polar(ksz)
                env = np.exp(-0.5 * (rho / env_sig) ** 2)
                # HealPixConv draws its (unused here) kernel from the global
                # RNG: fork it so that building a spectrum leaves the random
                # stream of a training script untouched
                with _forked_rng(self.device):
                    conv = HealPixConv(
                        level=j, in_channels=1, out_channels=1, kernel_sz=ksz,
                        n_gauges=self.G, gauge_type=gauge_type,
                        singularity_lonlat=singularity_lonlat, ref_direction=ref_direction,
                        cell_ids=ids, ellipsoid=ellipsoid, dtype=dtype, device=self.device,
                        cache_dir=cache_dir, missing="local_mean", local_mean_weights=env,
                    )
                with torch.no_grad():
                    for p in conv.parameters():
                        p.zero_()
                for p in conv.parameters():
                    p.requires_grad_(False)
                convs.append(conv)
                taps_l.append(taps)
                band_q.append(taps.shape[0])
            else:
                convs.append(None)
                taps_l.append(None)
                band_q.append(0)
                wide.append(True)
            if j >= 1:
                down = HealPixDown(level=j, mode=down_mode, ellipsoid=ellipsoid,
                                   cell_ids=ids, device=self.device, dtype=dtype)
                downs.append(down)
                # HealPixDown(mode="mean") divides by the number n_P of
                # children present in cell_ids.  Times n_P / 4, a coarse
                # value is the sum over its 4 children / 4, absent children
                # counting 0 (area-weighted): then sum_k q_j = sum_k q_J / 4^(J-j)
                # on a regional domain as on the full sphere, m stays in
                # [0, 1], and a pixel outside cell_ids counts like a NaN.
                area.append(torch.as_tensor(down.coverage, dtype=dtype)
                            if down_mode == "mean" and ids is not None else None)
                ids = None if ids is None else np.asarray(down.cell_ids_out, dtype=np.int64)
        self.convs = nn.ModuleList([c for c in convs if c is not None])
        self._conv_of_level = convs          # per level J..0, None without a band
        self.downs = nn.ModuleList(downs)
        self._area = []
        for i, a in enumerate(area):
            if a is None:
                self._area.append(None)
                continue
            self.register_buffer(f"_area_{i}", a, persistent=False)
            self._area.append(f"_area_{i}")

        # taps as buffers (complex for the anisotropic class)
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self._taps = []
        for i, taps in enumerate(taps_l):
            if taps is None:
                self._taps.append(None)
                continue
            if self._isotropic:
                t = torch.as_tensor(np.real(taps), dtype=dtype, device=self.device)
            else:
                t = torch.as_tensor(taps, dtype=cdtype, device=self.device)
            name = f"_taps_{i}"
            self.register_buffer(name, t, persistent=False)
            self._taps.append(name)

        # scale bookkeeping (fine to coarse)
        lev, voi, ell = [], [], []
        for j, nq, wd in zip(range(J, -1, -1), band_q, wide):
            for q in range(nq):
                lev.append(j)
                voi.append(q)
                kk = k0 if (wd and k0 is not None) else fallback.k0
                a = 2.0 ** (q / Q) if wd else 1.0
                ell.append(kk / a / (_ALPHA0 / 2 ** j))
        self.register_buffer("_levels", torch.tensor(lev), persistent=False)
        self.register_buffer("_voices", torch.tensor(voi), persistent=False)
        ell_t = torch.tensor(ell, dtype=torch.float64)
        self.register_buffer("_ell", ell_t, persistent=False)
        self.register_buffer("_scale_km", 2 * math.pi * _EARTH_RADIUS_KM / (ell_t + 0.5),
                             persistent=False)
        self.n_scale = len(lev)
        self.to(self.device)

    # ------------------------------------------------------------------
    def _prepare(self, x, weights):
        t = torch.as_tensor(x) if not torch.is_tensor(x) else x
        t = t.to(device=self.device, dtype=self.dtype)
        if t.dim() == 0:
            raise ValueError("x must have at least one dimension [..., N].")
        lead = tuple(t.shape[:-1])
        n = t.shape[-1]
        if n != self.n_cells[0]:
            raise ValueError(f"Expected {self.n_cells[0]} pixels, got {n}.")
        t = t.reshape(-1, n)
        w = torch.isfinite(t).to(self.dtype)
        if weights is not None:
            wt = torch.as_tensor(weights) if not torch.is_tensor(weights) else weights
            wt = wt.to(device=self.device, dtype=self.dtype)
            wt = torch.where(torch.isfinite(wt), wt, torch.zeros_like(wt)).clamp(0.0, 1.0)
            w = w * torch.broadcast_to(wt, lead + (n,)).reshape(-1, n)
        return t, w, lead

    def _down_step(self, di, *fields):
        """Take each ``[B, N_j]`` field one level down with ``self.downs[di]``.

        With ``down_mode="mean"`` on a regional domain, the output is
        multiplied by ``n_P / 4`` (area-weighted mean, see ``__init__``).
        """
        B = fields[0].shape[0]
        y, _ = self.downs[di](torch.cat(fields, dim=0))
        name = self._area[di]
        if name is not None:
            y = y * getattr(self, name)
        return y.split(B, dim=0)

    def _run(self, t, w):
        """Descent and bands: returns ([dict(y, d, u, j) per band level], M, P_low)."""
        q = w * torch.where(w > 0, t, torch.zeros_like(t))
        m = w
        # fraction of valid (w > 0) descendants, carried down like m: the
        # validity of a pixel does not depend on a weights= taper, which only
        # sets its share of the averages through u = m e
        f = (w > 0).to(self.dtype)
        # direct mean of the valid pixels: the monopole with "smooth" (which
        # does not conserve the mean); with "mean", the centring constant Mc,
        # corrected at level 0 below
        M, ok_m = _safe_div(q.sum(-1), m.sum(-1))          # [B]
        M = torch.where(ok_m, M, torch.full_like(M, float("nan")))
        Mc = torch.where(ok_m, M, torch.zeros_like(M))[:, None]
        # Centre before the descent: q/m is then the centred field at every
        # level (the descent is linear), with no float32 cancellation of a
        # field such as SST in kelvin at the coarse levels.
        q = w * torch.where(w > 0, t - Mc, torch.zeros_like(t))
        out = []
        di = 0
        for li, j in enumerate(range(self.level, -1, -1)):
            xj, _ = _safe_div(q, m)
            xj = torch.where(m > 0, xj, torch.zeros_like(xj))
            # at the input level f is binary: every pixel with w > 0 is valid
            vj = ((f >= self.min_valid) & (m > 0)).to(self.dtype)
            conv = self._conv_of_level[li]
            if conv is not None:
                taps = getattr(self, self._taps[li])                 # [Qj, P]
                z, cov = conv._local_mean_stencil(xj[:, None, :], vj[:, None, :])
                z, cov = z[:, 0], cov[:, 0]                           # [B, G, K, P]
                if self.edge == "none":
                    z = torch.where(cov > 1e-6, z, torch.zeros_like(z)) * cov
                if self._isotropic:
                    y = torch.einsum("bgkp,qp->bgqk", z, taps)
                else:
                    yr = torch.einsum("bgkp,qp->bgqk", z, taps.real)
                    yi = torch.einsum("bgkp,qp->bgqk", z, taps.imag)
                    y = torch.complex(yr, yi)
                a2 = taps.abs() ** 2                                   # [Qj, P]
                e = torch.einsum("bgkp,qp->bgqk", cov, a2) / a2.sum(-1)[None, None, :, None]
                base = (m * vj)[:, None, None, :]                      # [B, 1, 1, K]
                if self.edge == "mean_corrected":
                    d = e
                    u = base * (e if self.coverage_weight else torch.ones_like(e))
                    if self.min_coverage > 0:
                        u = u * (e >= self.min_coverage).to(self.dtype)
                    u = u * (e > 1e-6).to(self.dtype)
                elif self.edge == "complete":
                    a1 = taps.abs()
                    c = torch.einsum("bgkp,qp->bgqk", cov, a1) / a1.sum(-1)[None, None, :, None]
                    d = torch.ones_like(e)
                    u = base * (c >= 1.0 - 1e-5).to(self.dtype)
                else:
                    d = torch.ones_like(e)
                    u = base.expand_as(e)
                out.append({"y": y, "d": d, "u": u, "j": j})
            if j >= 1:
                q, m, f = self._down_step(di, q, m, f)
                di += 1
        # level 0: frequency 0 and low-pass residual around it
        x0, _ = _safe_div(q, m)                             # centred on Mc
        x0 = torch.where(m > 0, x0, torch.zeros_like(x0))
        if self.down_mode == "mean":
            # The area-weighted mean descent conserves the sums, so the
            # m-weighted mean of the level-0 pixels, sum q0 / sum m0 of the
            # uncentred pair, is the mean of the input.  With q centred on
            # Mc, it is Mc + dM; dM is 0 up to rounding.
            dM, _ = _safe_div(q.sum(-1), m.sum(-1))
            M = M + dM
            x0 = torch.where(m > 0, x0 - dM[:, None], torch.zeros_like(x0))
        P_low, ok_low = _safe_div((m * x0 ** 2).sum(-1), m.sum(-1))
        P_low = torch.where(ok_low & ok_m, P_low, torch.full_like(P_low, float("nan")))
        return out, M, P_low

    # ------------------------------------------------------------------
    def forward(self, x: ArrayLike, weights: Optional[ArrayLike] = None) -> WaveletSpectrum:
        """Band powers of ``x`` ([..., N]; NaN = missing; ``weights`` in [0, 1])."""
        t, w, lead = self._prepare(x, weights)
        bands, M, P_low = self._run(t, w)
        pw, s1, neff, maps = [], [], [], []
        nan = float("nan")
        for b in bands:
            y, d, u = b["y"], b["d"], b["u"]
            a2 = y.real ** 2 + y.imag ** 2 if torch.is_complex(y) else y ** 2
            p2, _ = _safe_div(a2, d)
            den = u.sum(-1)                                          # [B, G, Qj]
            P, ok = _safe_div((u * p2).sum(-1), den)
            pw.append(torch.where(ok, P, torch.full_like(P, nan)))
            neff.append(den.amin(1))                                 # [B, Qj], worst orientation
            if self.s1:
                ab = torch.abs(y)
                r, _ = _safe_div(ab, torch.sqrt(torch.clamp(d, min=0.0)))
                S, _ = _safe_div((u * r).sum(-1), den)
                s1.append(torch.where(ok, S, torch.full_like(S, nan)))
            if self.local_maps:
                B, G, Qj, K = p2.shape
                maps.append(p2.reshape(B, G * Qj, K).reshape(lead + (G * Qj, K)))
        power = torch.cat(pw, dim=2).transpose(1, 2)                 # [B, S, G]
        n_eff = torch.cat(neff, dim=1)                               # [B, S]
        power_total = power.sum(-1)
        s1_t = torch.cat(s1, dim=2).transpose(1, 2) if self.s1 else None
        S = power.shape[1]

        def shape(a, tail):
            return a.reshape(lead + tail)

        if self._isotropic:
            power_o = shape(power[..., 0], (S,))
            s1_o = None if s1_t is None else shape(s1_t[..., 0], (S,))
        else:
            power_o = shape(power, (S, self.G))
            s1_o = None if s1_t is None else shape(s1_t, (S, self.G))
        return WaveletSpectrum(
            power=power_o,
            power_total=shape(power_total, (S,)),
            s1=s1_o,
            monopole=M.reshape(lead),
            lowpass_var=P_low.reshape(lead),
            n_eff=shape(n_eff, (S,)),
            levels=self._levels,
            voices=self._voices,
            ell_nominal=self._ell,
            scale_km=self._scale_km,
            local_power=maps if self.local_maps else None,
        )

    # ------------------------------------------------------------------
    def cross(self, x: ArrayLike, y: Optional[ArrayLike] = None,
              weights_x: Optional[ArrayLike] = None,
              weights_y: Optional[ArrayLike] = None) -> WaveletCrossSpectrum:
        """Co-spectrum and coherence of two fields on the same cells.

        ``x`` and ``y`` have the same shape ``[..., N]`` (same cells, same
        order); their NaN masks may differ.  With ``y=None``, ``x`` must be
        ``[..., C, N]`` and every pair of the C fields is returned, with a
        ``[C, C]`` pair of axes after the leading dimensions.
        """
        if y is None:
            xt = torch.as_tensor(x) if not torch.is_tensor(x) else x
            if xt.dim() < 2:
                raise ValueError("with y=None, x must be [..., C, N].")
            C = xt.shape[-2]
            t, w, lead = self._prepare(xt, weights_x)                # lead = (..., C)
            outer = lead[:-1]
            bands, _, _ = self._run(t, w)
            return self._cross_reduce(bands, C, outer, pair=None)
        t1, w1, lead = self._prepare(x, weights_x)
        t2, w2, lead2 = self._prepare(y, weights_y)
        if lead2 != lead:
            raise ValueError("x and y must have the same shape.")
        B = t1.shape[0]
        bands, _, _ = self._run(torch.cat([t1, t2]), torch.cat([w1, w2]))
        return self._cross_reduce(bands, 2, lead, pair=(B,))

    def _cross_reduce(self, bands, C, lead, pair):
        co, quad, px, py, neff = [], [], [], [], []
        nan = float("nan")
        for b in bands:
            y, d, u = b["y"], b["d"], b["u"]
            if pair is not None:
                B = pair[0]
                ya, yb = y[:B, None], y[B:, None]                    # [B, 1, G, Q, K]
                da, db = d[:B, None], d[B:, None]
                ua, ub = u[:B, None], u[B:, None]
            else:
                shp = (-1, C) + tuple(y.shape[1:])
                y_, d_, u_ = y.reshape(shp), d.reshape(shp), u.reshape(shp)
                ya, yb = y_[:, :, None], y_[:, None, :]              # [B, C, C, G, Q, K]
                da, db = d_[:, :, None], d_[:, None, :]
                ua, ub = u_[:, :, None], u_[:, None, :]
            # joint weight: geometric mean, so that cross(x, x) is the auto
            # spectrum; for binary masks it is the intersection
            prod_u = ua * ub
            pos_u = prod_u > 0
            uab = torch.where(pos_u, torch.sqrt(torch.where(pos_u, prod_u, torch.ones_like(prod_u))),
                              torch.zeros_like(prod_u))
            den = uab.sum(-1)
            sd, _ = _safe_div(torch.ones_like(da * db), torch.sqrt(da * db))
            if torch.is_complex(y):
                prod = ya * torch.conj(yb)
                cre, cim = prod.real * sd, prod.imag * sd
            else:
                cre, cim = ya * yb * sd, None
            ia, _ = _safe_div(torch.ones_like(da), da)
            ib, _ = _safe_div(torch.ones_like(db), db)
            pa = (ya.real ** 2 + ya.imag ** 2 if torch.is_complex(ya) else ya ** 2) * ia
            pb = (yb.real ** 2 + yb.imag ** 2 if torch.is_complex(yb) else yb ** 2) * ib
            c, ok = _safe_div((uab * cre).sum(-1), den)
            co.append(torch.where(ok, c, torch.full_like(c, nan)))
            if cim is not None:
                qd, _ = _safe_div((uab * cim).sum(-1), den)
                quad.append(torch.where(ok, qd, torch.full_like(qd, nan)))
            a_, _ = _safe_div((uab * pa).sum(-1), den)
            b_, _ = _safe_div((uab * pb).sum(-1), den)
            px.append(torch.where(ok, a_, torch.full_like(a_, nan)))
            py.append(torch.where(ok, b_, torch.full_like(b_, nan)))
            neff.append(den.amin(-2))
        cat = lambda lst: torch.cat(lst, dim=-1).transpose(-1, -2)   # [..., S, G]
        co_g, px_g, py_g = cat(co), cat(px), cat(py)
        co_t, px_t, py_t = co_g.sum(-1), px_g.sum(-1), py_g.sum(-1)
        coh, ok = _safe_div(co_t, torch.sqrt(px_t * py_t))
        coh = torch.where(ok, coh, torch.full_like(coh, nan))
        n_eff = torch.cat(neff, dim=-1)
        S = co_t.shape[-1]
        if pair is not None:
            tail = lambda a: a[:, 0]
            pre = lead
        else:
            tail = lambda a: a
            pre = lead + (C, C)
        r = lambda a, extra: tail(a).reshape(pre + extra)
        quad_o = None
        if self._isotropic:
            co_o = r(co_g[..., 0], (S,))
        else:
            co_o = r(co_g, (S, self.G))
            quad_o = r(cat(quad), (S, self.G))
        return WaveletCrossSpectrum(
            co=co_o, quad=quad_o, co_total=r(co_t, (S,)), coherence=r(coh, (S,)),
            power_x=r(px_t, (S,)), power_y=r(py_t, (S,)), n_eff=r(n_eff, (S,)),
            levels=self._levels, voices=self._voices,
        )

    # ------------------------------------------------------------------
    def windows(self, lmax: Optional[int] = None, batch: int = 256) -> WaveletWindows:
        """Measure the response of each scale to unit-power multipoles.

        Full sphere only.  For every ell, the 2 ell + 1 real orthonormal
        harmonics are synthesised with ``HEALPixSHT.alm2map`` (NESTED) and
        passed through the spectrum: the sum of their band powers is the
        response.  Exact for a rotation-invariant operator; ours is invariant
        up to the pixelisation.  Costs (lmax + 1)^2 maps and the Legendre
        tables of HEALPixSHT (about 36 nside^3 values): use small levels.
        """
        if self.partial:
            raise ValueError("windows() needs the full sphere (cell_ids=None).")
        from healpix_analyse.healpix_sht import HEALPixSHT

        sht = HEALPixSHT(self.level, lmax=lmax, dtype=torch.float64, device="cpu")
        L = sht.lmax
        ell_idx, m_idx = [], []
        for m in range(L + 1):
            for ell in range(m, L + 1):
                ell_idx.append(ell)
                m_idx.append(m)
        ell_idx = np.asarray(ell_idx)
        m_idx = np.asarray(m_idx)
        n_alm = len(ell_idx)
        rows, row_ell = [], []
        for k in range(n_alm):
            if m_idx[k] == 0:
                a = np.zeros(n_alm, complex)
                a[k] = 1.0
                rows.append(a)
                row_ell.append(ell_idx[k])
            else:
                for val in (1.0 / math.sqrt(2.0), 1j / math.sqrt(2.0)):
                    a = np.zeros(n_alm, complex)
                    a[k] = val
                    rows.append(a)
                    row_ell.append(ell_idx[k])
        row_ell = torch.as_tensor(np.asarray(row_ell))
        resp = torch.zeros(self.n_scale, L + 1, dtype=torch.float64)
        low = torch.zeros(L + 1, dtype=torch.float64)
        with torch.no_grad():
            for s0 in range(0, len(rows), batch):
                alm = torch.as_tensor(np.stack(rows[s0:s0 + batch]))
                maps = sht.alm2map(alm, nest=True).real
                out = self.forward(maps.to(self.dtype))
                pt = out.power_total.to(torch.float64).cpu()         # [b, S]
                le = row_ell[s0:s0 + batch]
                resp.index_add_(1, le, pt.T)
                low.index_add_(0, le, out.lowpass_var.to(torch.float64).cpu())
        ell = torch.arange(L + 1, dtype=torch.float64)
        wts = resp / (2 * ell + 1)
        ell_eff = (wts * ell).sum(-1) / wts.sum(-1)
        return WaveletWindows(ell=ell, response=resp, lowpass=low, ell_eff=ell_eff)

    def littlewood_paley(self, windows: Optional[WaveletWindows] = None,
                         lmax: Optional[int] = None) -> LittlewoodPaley:
        """Fit one constant per band and one for the low-pass (least squares).

        The variance of a full-sphere map around its mean is
        ``sum_{ell >= 1} (2 ell + 1) / (4 pi) C_ell``; the constants make
        ``sum_s c_s response[s, ell] + c_low lowpass[ell]`` as close as
        possible to ``(2 ell + 1) / (4 pi)`` for every ell >= 1, relative to
        it.  Uses :meth:`windows` (full sphere, small levels).
        """
        win = self.windows(lmax) if windows is None else windows
        ell = win.ell
        sel = ell >= 1
        norm = (4 * math.pi / (2 * ell + 1))[sel]
        A = torch.cat([(win.response[:, sel] * norm).T,
                       (win.lowpass[sel] * norm)[:, None]], dim=1)
        rhs = torch.ones(A.shape[0], 1, dtype=A.dtype)
        c = torch.linalg.lstsq(A, rhs).solution[:, 0]
        fit = A @ c
        ripple = (fit.max() - fit.min()) / fit.mean()
        return LittlewoodPaley(constants=c[:-1], lowpass=c[-1], ripple=ripple)


# ===========================================================================
# Public classes and function
# ===========================================================================

class HealPixWaveletSpectrum(_WaveletSpectrumBase):
    """Wavelet power spectrum of HEALPix maps, oriented (complex Morlet, G gauges).

    Parameters
    ----------
    level : int
        Level of the input maps (``nside = 2**level``), at least 1.
    cell_ids : array-like or None
        NESTED cell ids of a regional domain; None = full sphere.
    n_scales : int, default -1
        Number of levels with a band, from ``level`` down; -1 = all, down to
        level 0.  Otherwise at least 1.  The descent always goes to level 0.
    wavelet : callable or None
        Profile ``fn(rho_pix, phi_rad)``; None = :func:`wavelet_morlet` with
        the defaults of ``scales_per_octave``.
    scales_per_octave : {1, 2}, default 1
        Q scales per octave: Q wavelets dilated by 2**(q/Q) at each level.
        With the default wavelet, Q = 2 also makes each band half as wide.
    kernel_sz : int or None
        None = derived from the widest voice (5 for Q = 1, 11 for Q = 2).
        A value that cuts more than 0.1 % of the envelope energy raises.
    n_gauges : int or None
        Orientations; None = 4 Q.
    gauge_type, singularity_lonlat, ref_direction
        As in :class:`~healpix_analyse.convol.HealPixConv`.
    down_mode : {"mean", "smooth"}, default "mean"
        ``HealPixDown`` mode of the descent of ``(q, m) = (w (x - M), w)``.
        ``"mean"``: the mean of the 4 children, area-weighted on a regional
        domain (a child outside ``cell_ids`` counts 0, like a NaN), which
        conserves the sums: the monopole is the ``m``-weighted mean of the
        level-0 pixels, ``sum q_0 / sum m_0``, equal to the mean of the
        valid input pixels.  The boxcar aliases: below ``level``, a band also
        picks up multipoles beyond the grid of its level (about half of the
        power for white noise, 5 to 13 % for a C_ell ~ ell^-3 spectrum).
        ``"smooth"``: the Gaussian descent, which aliases about 1 % but does
        not conserve the mean; the monopole is then computed directly from
        the input.  Only these two keep ``q / m`` a weighted mean of the
        valid pixels; the other modes raise (``maxpool`` lets a missing
        child win, ``smooth_conservative`` biases ``q / m`` near coasts).
    edge : {"mean_corrected", "complete", "none"}, default "mean_corrected"
        Missing stencil points: completed with the local mean (``|y|^2`` divided
        by the coverage e); outputs with an incomplete kernel dropped; or set
        to the frequency 0.
    coverage_weight : bool, default True
        With ``mean_corrected``: weight every valid pixel by its coverage e
        in the mean, so that no valid pixel is left out.
    min_coverage : float, default 0.0
        With ``mean_corrected``: drop the pixels whose coverage e is below.
    min_valid : float, default 0.5
        A coarse pixel is valid when the fraction of its descendants with a
        positive weight (carried down like m; with ``"mean"``, a descendant
        outside ``cell_ids`` counts as missing) is at least this; at ``level``
        itself, every pixel with a positive weight is valid.  A ``weights=``
        taper never makes a pixel invalid: it only sets the share of each
        valid pixel in the averages (``u = m e``).
    s1 : bool, default False
        Also return S1, the weighted mean of ``|y|``.
    local_maps : bool, default False
        Also return the ``|y|^2`` maps of every level.
    max_stencil_deg : float, default 180.0
        At levels where the corners of the stencil would be farther than
        this from the centre, the band uses the default 5x5 wavelet.
    ellipsoid, dtype, device, cache_dir
        As in ``HealPixConv``.
    """

    _isotropic = False

    def __init__(
        self,
        level: int,
        cell_ids: Optional[ArrayLike] = None,
        n_scales: int = -1,
        wavelet: Optional[KernelFn] = None,
        scales_per_octave: int = 1,
        kernel_sz: Optional[int] = None,
        n_gauges: Optional[int] = None,
        gauge_type: str = "phi",
        singularity_lonlat=None,
        ref_direction=None,
        down_mode: str = "mean",
        edge: str = "mean_corrected",
        coverage_weight: bool = True,
        min_coverage: float = 0.0,
        min_valid: float = 0.5,
        s1: bool = False,
        local_maps: bool = False,
        max_stencil_deg: float = 180.0,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float32,
        device=None,
        cache_dir=_DEFAULT_CACHE_DIR,
    ):
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        super().__init__(
            level, cell_ids, n_scales, wavelet, scales_per_octave, kernel_sz,
            4 * int(scales_per_octave) if n_gauges is None else int(n_gauges),
            gauge_type, singularity_lonlat, ref_direction, down_mode, edge,
            coverage_weight, min_coverage, min_valid, s1, local_maps,
            max_stencil_deg, ellipsoid, dtype, device, cache_dir,
        )


class HealPixIsotropicWaveletSpectrum(_WaveletSpectrumBase):
    """Isotropic wavelet power spectrum (one real radial wavelet, no gauges).

    Same parameters as :class:`HealPixWaveletSpectrum` without the gauge
    options; ``wavelet`` is a radial profile ``fn(rho_pix, phi_rad)``
    (default :func:`wavelet_morlet_isotropic`, 7x7 for Q = 1).  About 4x
    less memory than the oriented class at the same kernel size, and one
    real output per pixel instead of 2 G.
    """

    _isotropic = True

    def __init__(
        self,
        level: int,
        cell_ids: Optional[ArrayLike] = None,
        n_scales: int = -1,
        wavelet: Optional[KernelFn] = None,
        scales_per_octave: int = 1,
        kernel_sz: Optional[int] = None,
        down_mode: str = "mean",
        edge: str = "mean_corrected",
        coverage_weight: bool = True,
        min_coverage: float = 0.0,
        min_valid: float = 0.5,
        s1: bool = False,
        local_maps: bool = False,
        max_stencil_deg: float = 180.0,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float32,
        device=None,
        cache_dir=_DEFAULT_CACHE_DIR,
    ):
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        super().__init__(
            level, cell_ids, n_scales, wavelet, scales_per_octave, kernel_sz,
            1, "phi", None, None, down_mode, edge, coverage_weight, min_coverage,
            min_valid, s1, local_maps, max_stencil_deg, ellipsoid, dtype, device,
            cache_dir,
        )


def wavelet_power_spectrum(x: ArrayLike, level: int, cell_ids: Optional[ArrayLike] = None,
                           weights: Optional[ArrayLike] = None, isotropic: bool = False,
                           **kwargs) -> WaveletSpectrum:
    """One-call wavelet power spectrum: builds the operator and applies it.

    ``isotropic=True`` uses :class:`HealPixIsotropicWaveletSpectrum`, else
    :class:`HealPixWaveletSpectrum`; ``kwargs`` go to the class.  Build the
    class once and reuse it when you have several maps.
    """
    cls = HealPixIsotropicWaveletSpectrum if isotropic else HealPixWaveletSpectrum
    return cls(level, cell_ids=cell_ids, **kwargs)(x, weights=weights)


__all__ = [
    "HealPixWaveletSpectrum",
    "HealPixIsotropicWaveletSpectrum",
    "wavelet_power_spectrum",
    "wavelet_morlet",
    "wavelet_morlet_isotropic",
    "WaveletSpectrum",
    "WaveletCrossSpectrum",
    "WaveletWindows",
    "LittlewoodPaley",
]
