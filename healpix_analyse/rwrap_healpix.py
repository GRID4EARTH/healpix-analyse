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
from healpix_analyse.down import HealPixDown

try:
    import healpix_geo as hg
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "rwrap_healpix requires the 'healpix_geo' package. "
        "Install it with: pip install healpix-geo"
    ) from exc

ArrayLike = Union[np.ndarray, torch.Tensor]


class HealPixUp(nn.Module):
    """
    HEALPix ×2 upsampling via nearest-neighbour interpolation (NESTED ordering).
    """

    def __init__(self, nside_out: int, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.nside_out = int(nside_out)
        self.nside_in = self.nside_out // 2
        self._cell_ids_out = np.arange(12 * self.nside_out ** 2, dtype=np.int64)

    @property
    def cell_ids_out(self) -> np.ndarray:
        return self._cell_ids_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        x = x.to(self.device)
        return x.repeat_interleave(4, dim=-1), self._cell_ids_out


class HealPixDoubleConv(nn.Module):
    """
    Two consecutive HealPixConv layers, each followed by GroupNorm and ReLU.
    """

    def __init__(
        self,
        nside: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[Any] = None,
        ellipsoid: str = "WGS84",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        mid_ch = n_gauges * out_channels
        self.conv1 = HealPixConv(
            nside, in_channels, out_channels,
            kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ellipsoid=ellipsoid, device=device, dtype=dtype,
        )
        self.norm1 = nn.GroupNorm(_find_groups(mid_ch), mid_ch)
        self.conv2 = HealPixConv(
            nside, mid_ch, out_channels,
            kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ellipsoid=ellipsoid, device=device, dtype=dtype,
        )
        self.norm2 = nn.GroupNorm(_find_groups(mid_ch), mid_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


def _find_groups(num_channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return max(g, 1)


class HealPixGradient(nn.Module):
    """
    Fixed least-squares tangent-plane gradient operator on the HEALPix sphere.

    For each pixel, a local gradient is estimated from the kth-ring neighbourhood
    using a weighted least-squares fit on the tangent plane:

        T(j) - T(i) ≈ dT/deast * dx_ij + dT/dnorth * dy_ij

    with:
        dx_ij = cos(lat_i) * dlon_ij  [radians]
        dy_ij = dlat_ij               [radians]

    The pseudo-inverse weights are fully precomputed at construction time.
    The forward pass is differentiable with respect to the input field.
    """

    def __init__(
        self,
        nside: int,
        *,
        ring: int = 1,
        ellipsoid: str = "WGS84",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.nside = int(nside)
        self.depth = int(round(math.log2(self.nside)))
        self.ring = int(ring)
        self.ellipsoid = str(ellipsoid)

        npix = 12 * self.nside ** 2
        ipix = np.arange(npix, dtype=np.uint64)

        lon_deg, lat_deg = hg.nested.healpix_to_lonlat(ipix, self.depth, ellipsoid=self.ellipsoid)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)

        neigh = hg.nested.kth_neighbourhood(ipix, self.depth, self.ring)
        neigh = np.asarray(neigh, dtype=np.int64)
        if neigh.ndim != 2:
            raise ValueError(f"Expected a 2-D neighbourhood array, got shape={neigh.shape}.")

        valid = neigh >= 0
        center = ipix.astype(np.int64)[:, None]
        neigh_safe = np.where(valid, neigh, center)

        lon_i = lon[:, None]
        lat_i = lat[:, None]
        lon_j = lon[neigh_safe]
        lat_j = lat[neigh_safe]

        dlon = ((lon_j - lon_i + 180.0) % 360.0) - 180.0
        dlon = np.deg2rad(dlon)
        dlat = np.deg2rad(lat_j - lat_i)
        dx = dlon * np.cos(np.deg2rad(lat_i))
        dy = dlat

        # Avoid using the center pixel in the fit if returned by kth_neighbourhood.
        is_center = neigh_safe == center
        valid = valid & (~is_center)

        dx = np.where(valid, dx, 0.0)
        dy = np.where(valid, dy, 0.0)
        dist2 = dx * dx + dy * dy
        w = np.where(valid, 1.0 / (dist2 + 1e-8), 0.0)

        m = neigh_safe.shape[1]
        pinv = np.zeros((npix, 2, m), dtype=np.float64)
        eye2 = np.eye(2, dtype=np.float64)

        for k in range(npix):
            A = np.stack([dx[k], dy[k]], axis=1)          # [M, 2]
            W = np.diag(w[k])                             # [M, M]
            ATA = A.T @ W @ A                             # [2, 2]
            ATA += 1e-8 * eye2
            ATW = A.T @ W                                 # [2, M]
            pinv[k] = np.linalg.solve(ATA, ATW)

        self.register_buffer("neighbor_idx", torch.as_tensor(neigh_safe, dtype=torch.long))
        self.register_buffer("valid_mask", torch.as_tensor(valid.astype(np.float32), dtype=dtype))
        self.register_buffer("pinv", torch.as_tensor(pinv, dtype=dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor [B, C, K]

        Returns
        -------
        grad_east  : torch.Tensor [B, C, K]
        grad_north : torch.Tensor [B, C, K]
        """
        x = x.to(self.device)
        B, C, K = x.shape
        idx = self.neighbor_idx.view(1, 1, K, -1).expand(B, C, K, -1)
        x_src = x.unsqueeze(-1).expand(-1, -1, -1, idx.shape[-1])
        neigh = torch.gather(x_src, 2, idx)
        diff = (neigh - x.unsqueeze(-1)) * self.valid_mask.view(1, 1, K, -1)
        grad = torch.einsum("kdm,bckm->bcdk", self.pinv, diff)
        return grad[:, :, 0, :], grad[:, :, 1, :]


class HealPixAdvectionResidualNet(nn.Module):
    """
    Gauge-equivariant spherical warp+residual forecaster for HEALPix maps.

    The network predicts, for each forecast horizon h:
      - a tangent-plane eastward velocity u_h
      - a tangent-plane northward velocity v_h
      - a residual increment r_h

    Given the current SST T_h, the advection increment is approximated by the
    first-order transport equation on the sphere:

        dT_adv = -(u_h * dT/deast + v_h * dT/dnorth)

    and the final increment is:

        dT_h = gate_scale * clamp(dT_adv + r_h, ±residual_clip)

    Then either:
      - output_mode="delta"      → return dT_h
      - output_mode="cumulative" → return T_{h+1} = T_h + dT_h

    The advection step is explicit and recursive: the gradients of T_h are
    recomputed at each horizon.
    """

    def __init__(
        self,
        nside: int,
        in_channels: int,
        out_channels: int,
        feature_channels: List[int],
        *,
        vars_per_t: int = 1,
        time_steps: int = 1,
        down_mode: str = "smooth",
        weight_norm: str = "l1",
        residual_clip: float = 4.0,
        gate_scale: float = 1.0,
        tau: float = 4.0,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[Any] = None,
        ellipsoid: str = "WGS84",
        grad_ring: int = 1,
        smoothness_lambda: float = 0.05,
        residual_lambda: float = 0.01,
        output_mode: str = "delta",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype

        if len(feature_channels) < 1:
            raise ValueError("feature_channels must have at least one level.")
        if output_mode not in {"delta", "cumulative"}:
            raise ValueError("output_mode must be 'delta' or 'cumulative'.")

        self.nside = int(nside)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_channels = list(feature_channels)
        self.vars_per_t = int(vars_per_t)
        self.time_steps = int(time_steps)
        self.down_mode = str(down_mode)
        self.weight_norm = str(weight_norm)
        self.residual_clip = float(residual_clip)
        self.gate_scale = float(gate_scale)
        self.tau = float(tau)
        self.kernel_sz = int(kernel_sz)
        self.n_gauges = int(n_gauges)
        self.gauge_type = str(gauge_type)
        self.singularity_lonlat = singularity_lonlat
        self.ellipsoid = str(ellipsoid)
        self.grad_ring = int(grad_ring)
        self.smoothness_lambda = float(smoothness_lambda)
        self.residual_lambda = float(residual_lambda)
        self.output_mode = str(output_mode)
        self.sst_index = self.vars_per_t - 1

        expected_ci = self.time_steps * self.vars_per_t + 1
        if self.in_channels != expected_ci:
            import warnings
            warnings.warn(
                f"in_channels={self.in_channels} != time_steps*vars_per_t+1={expected_ci}. "
                "Make sure your input encoding matches the expected layout.",
                UserWarning, stacklevel=2,
            )

        self.nside_levels: List[int] = [self.nside]
        tmp = self.nside
        for _ in range(len(self.feature_channels) - 1):
            tmp //= 2
            if tmp < 1:
                raise ValueError("Too many levels in feature_channels for the given nside.")
            self.nside_levels.append(tmp)
        L = len(self.feature_channels) - 1

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

        self.enc_convs = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        self.enc_convs.append(
            HealPixDoubleConv(**_dconv_kwargs(self.nside_levels[0], self.in_channels, self.feature_channels[0]))
        )
        for i in range(L):
            self.down_ops.append(
                HealPixDown(
                    nside_in=self.nside_levels[i],
                    mode=self.down_mode,
                    weight_norm=self.weight_norm,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            in_ch = self.n_gauges * self.feature_channels[i]
            self.enc_convs.append(
                HealPixDoubleConv(**_dconv_kwargs(self.nside_levels[i + 1], in_ch, self.feature_channels[i + 1]))
            )

        self.up_ops = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for j in range(L):
            enc_lvl = L - 1 - j
            fine_nside = self.nside_levels[enc_lvl]
            self.up_ops.append(HealPixUp(nside_out=fine_nside, device=self.device))
            in_ch = self.n_gauges * (self.feature_channels[enc_lvl + 1] + self.feature_channels[enc_lvl])
            out_ch = self.feature_channels[enc_lvl]
            self.dec_convs.append(HealPixDoubleConv(**_dconv_kwargs(fine_nside, in_ch, out_ch)))

        final_ch = self.n_gauges * self.feature_channels[0]
        self.flow_head = nn.Conv1d(final_ch, 2 * self.out_channels, kernel_size=1)
        self.res_head = nn.Conv1d(final_ch, self.out_channels, kernel_size=1)
        self.grad_op = HealPixGradient(
            self.nside,
            ring=self.grad_ring,
            ellipsoid=self.ellipsoid,
            device=self.device,
            dtype=self.dtype,
        )

        self._last_flow_east: Optional[torch.Tensor] = None
        self._last_flow_north: Optional[torch.Tensor] = None
        self._last_residual: Optional[torch.Tensor] = None
        self._last_delta: Optional[torch.Tensor] = None

        self.to(self.device)

    @staticmethod
    def _apply_down(down_op: HealPixDown, feat: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        B, C, K = feat.shape
        y, cell_ids = down_op(feat.reshape(B * C, K))
        K_out = y.shape[-1]
        return y.reshape(B, C, K_out), cell_ids

    @staticmethod
    def _corrcoef_1d(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        a0 = a - a.mean(dim=dim, keepdim=True)
        b0 = b - b.mean(dim=dim, keepdim=True)
        num = (a0 * b0).sum(dim=dim)
        den = torch.sqrt((a0 ** 2).sum(dim=dim) * (b0 ** 2).sum(dim=dim) + eps)
        return num / den

    def _temporal_weights(self, T_out: int, device: torch.device) -> torch.Tensor:
        k = torch.arange(T_out, device=device, dtype=torch.float32)
        w = torch.exp(-k / self.tau)
        return w / (w.sum() + 1e-12)

    def loss_mse_temporal(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        B, T_out, K = y_pred.shape
        err = (y_pred - y_true) ** 2
        w = self._temporal_weights(T_out, y_pred.device)
        return (w.view(1, T_out, 1) * err).mean()

    def loss_corr_spatial(self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        corr_bt = self._corrcoef_1d(y_pred, y_true, dim=-1, eps=eps)
        w = self._temporal_weights(y_pred.shape[1], y_pred.device)
        return 1.0 - (corr_bt * w.unsqueeze(0)).sum(dim=1).mean()

    def loss_corr_plus_zmse(self, y_pred: torch.Tensor, y_true: torch.Tensor, lam: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
        lc = self.loss_corr_spatial(y_pred, y_true, eps=eps)
        mu_p = y_pred.mean(dim=-1, keepdim=True)
        mu_t = y_true.mean(dim=-1, keepdim=True)
        std_p = y_pred.std(dim=-1, keepdim=True) + eps
        std_t = y_true.std(dim=-1, keepdim=True) + eps
        zp = (y_pred - mu_p) / std_p
        zt = (y_true - mu_t) / std_t
        zmse = (zp - zt).pow(2).mean()
        return lc + lam * zmse

    def flow_smoothness_loss(
        self,
        flow_east: Optional[torch.Tensor] = None,
        flow_north: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flow_east is None:
            flow_east = self._last_flow_east
        if flow_north is None:
            flow_north = self._last_flow_north
        if flow_east is None or flow_north is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        ge_x, gn_x = self.grad_op(flow_east)
        ge_y, gn_y = self.grad_op(flow_north)
        return (ge_x.pow(2).mean() + gn_x.pow(2).mean() + ge_y.pow(2).mean() + gn_y.pow(2).mean())

    def residual_l1_loss(self, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is None:
            residual = self._last_residual
        if residual is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return residual.abs().mean()

    def total_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base = self.loss_mse_temporal(y_pred, y_true)
        return base + self.smoothness_lambda * self.flow_smoothness_loss() + self.residual_lambda * self.residual_l1_loss()

    def temporal_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.total_loss(y_pred, y_true)

    def _decode_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        enc_feats: List[torch.Tensor] = []
        feat = self.enc_convs[0](feat)
        enc_feats.append(feat)
        for level in range(len(self.feature_channels) - 1):
            feat, _ = self._apply_down(self.down_ops[level], feat)
            feat = self.enc_convs[level + 1](feat)
            enc_feats.append(feat)
        x_dec = enc_feats[-1]
        for j in range(len(self.up_ops)):
            x_up, _ = self.up_ops[j](x_dec)
            skip = enc_feats[-(j + 2)]
            x_cat = torch.cat([x_up, skip], dim=1)
            x_dec = self.dec_convs[j](x_cat)
        return x_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        B, C, K = x.shape
        N_full = 12 * self.nside ** 2
        if K != N_full:
            raise ValueError(f"Expected full-sphere K={N_full} pixels at nside={self.nside}, got K={K}.")

        phys = x[:, :-1, :]
        T, V = self.time_steps, self.vars_per_t
        if phys.shape[1] != T * V:
            raise ValueError(f"Expected phys channels T*V={T*V}, got {phys.shape[1]}.")
        phys_4d = phys.view(B, T, V, K)
        current = phys_4d[:, -1, self.sst_index, :].unsqueeze(1)

        x_dec = self._decode_features(x)
        flow = self.flow_head(x_dec).view(B, self.out_channels, 2, K)
        residual = self.res_head(x_dec)

        flow_east = flow[:, :, 0, :]
        flow_north = flow[:, :, 1, :]

        deltas: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        current_state = current
        for h in range(self.out_channels):
            grad_e, grad_n = self.grad_op(current_state)
            adv_delta = -(flow_east[:, h:h+1, :] * grad_e + flow_north[:, h:h+1, :] * grad_n)
            delta = torch.clamp(adv_delta + residual[:, h:h+1, :], -self.residual_clip, self.residual_clip)
            delta = delta * self.gate_scale
            deltas.append(delta)
            current_state = current_state + delta
            outputs.append(current_state)

        self._last_flow_east = flow_east
        self._last_flow_north = flow_north
        self._last_residual = residual
        self._last_delta = torch.cat(deltas, dim=1)

        if self.output_mode == "delta":
            return self._last_delta
        return torch.cat(outputs, dim=1)

    def predict_components(self, x: ArrayLike, batch_size: int = 16) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        x = x.float()

        flow_e, flow_n, residual, delta = [], [], [], []
        self.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = x[start:start + batch_size].to(self.device)
                _ = self(xb)
                flow_e.append(self._last_flow_east.cpu())
                flow_n.append(self._last_flow_north.cpu())
                residual.append(self._last_residual.cpu())
                delta.append(self._last_delta.cpu())
        return {
            "flow_east": torch.cat(flow_e, dim=0),
            "flow_north": torch.cat(flow_n, dim=0),
            "residual": torch.cat(residual, dim=0),
            "delta": torch.cat(delta, dim=0),
        }

    def fit(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        n_epoch: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        optimizer: str = "adam",
        view_epoch: int = 10,
        loss_fn: Optional[Any] = None,
    ) -> Dict[str, List]:
        loss_fn = loss_fn or self.temporal_loss

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

        opt_name = optimizer.lower().strip()
        if opt_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "lbfgs":
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20, history_size=10)
        else:
            raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got '{optimizer}'.")

        history: Dict[str, List] = {"train_loss": [], "val_loss": []}
        for epoch in range(1, n_epoch + 1):
            self.train()
            if opt_name == "adam":
                perm = torch.randperm(N, device=self.device)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, N, batch_size):
                    idx = perm[start:start + batch_size]
                    xb, yb = x_train[idx], y_train[idx]
                    opt.zero_grad()
                    loss = loss_fn(self(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    opt.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                train_loss = epoch_loss / max(n_batches, 1)
            else:
                def closure():
                    opt.zero_grad()
                    loss = loss_fn(self(x_train), y_train)
                    loss.backward()
                    return loss
                train_loss = float(opt.step(closure).item())

            val_loss = None
            if has_val:
                self.eval()
                with torch.no_grad():
                    Nv = x_val.shape[0]
                    if Nv <= batch_size or opt_name == "lbfgs":
                        val_loss = float(loss_fn(self(x_val), y_val).item())
                    else:
                        acc, nb = 0.0, 0
                        for start in range(0, Nv, batch_size):
                            xb = x_val[start:start + batch_size]
                            yb = y_val[start:start + batch_size]
                            acc += float(loss_fn(self(xb), yb).item())
                            nb += 1
                        val_loss = acc / max(nb, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if epoch == 1 or epoch % view_epoch == 0 or epoch == n_epoch:
                if val_loss is not None:
                    print(f"[Epoch {epoch:4d}/{n_epoch}]  train={train_loss:.6f}  val={val_loss:.6f}")
                else:
                    print(f"[Epoch {epoch:4d}/{n_epoch}]  train={train_loss:.6f}")
        return history

    def predict(self, x: ArrayLike, batch_size: int = 16) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        x = x.float()
        self.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = x[start:start + batch_size].to(self.device)
                preds.append(self(xb).cpu())
        return torch.cat(preds, dim=0)

    def get_hparams(self) -> Dict[str, Any]:
        return {
            "nside": self.nside,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "feature_channels": list(self.feature_channels),
            "vars_per_t": self.vars_per_t,
            "time_steps": self.time_steps,
            "down_mode": self.down_mode,
            "weight_norm": self.weight_norm,
            "residual_clip": self.residual_clip,
            "gate_scale": self.gate_scale,
            "tau": self.tau,
            "kernel_sz": self.kernel_sz,
            "n_gauges": self.n_gauges,
            "gauge_type": self.gauge_type,
            "singularity_lonlat": self.singularity_lonlat,
            "ellipsoid": self.ellipsoid,
            "grad_ring": self.grad_ring,
            "smoothness_lambda": self.smoothness_lambda,
            "residual_lambda": self.residual_lambda,
            "output_mode": self.output_mode,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        fn = (lambda p: p.requires_grad) if trainable_only else (lambda p: True)
        return sum(p.numel() for p in self.parameters() if fn(p))

    def parameter_table(self) -> Dict[str, Any]:
        rows, total, trainable = [], 0, 0
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            trainable += n if p.requires_grad else 0
            rows.append({"name": name, "shape": list(p.shape), "numel": int(n), "trainable": bool(p.requires_grad)})
        return {"total": int(total), "trainable": int(trainable), "tensors": rows}

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [HealPixAdvectionResidualNet._json_safe(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): HealPixAdvectionResidualNet._json_safe(v) for k, v in obj.items()}
        return str(obj)

    def save_checkpoint(
        self,
        path: str,
        *,
        history: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        save_json_sidecar: bool = True,
    ) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_class": self.__class__.__name__,
            "hparams": self.get_hparams(),
            "model_state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "epoch": epoch,
            "history": history,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "param_report": self.parameter_table(),
            "versions": {"torch": torch.__version__, "numpy": np.__version__},
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": extra or {},
        }
        torch.save(ckpt, path)
        if save_json_sidecar:
            meta = {k: ckpt[k] for k in ("timestamp", "model_class", "hparams", "epoch", "history", "versions", "extra")}
            meta["param_report"] = {"total": ckpt["param_report"]["total"], "trainable": ckpt["param_report"]["trainable"]}
            json_path = os.path.splitext(path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: Optional[torch.device] = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["HealPixAdvectionResidualNet", Dict[str, Any]]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        h = ckpt["hparams"].copy()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(
            nside=h["nside"],
            in_channels=h["in_channels"],
            out_channels=h["out_channels"],
            feature_channels=h["feature_channels"],
            vars_per_t=h.get("vars_per_t", 1),
            time_steps=h.get("time_steps", 1),
            down_mode=h.get("down_mode", "smooth"),
            weight_norm=h.get("weight_norm", "l1"),
            residual_clip=h.get("residual_clip", 4.0),
            gate_scale=h.get("gate_scale", 1.0),
            tau=h.get("tau", 4.0),
            kernel_sz=h.get("kernel_sz", 3),
            n_gauges=h.get("n_gauges", 1),
            gauge_type=h.get("gauge_type", "phi"),
            singularity_lonlat=h.get("singularity_lonlat", None),
            ellipsoid=h.get("ellipsoid", "WGS84"),
            grad_ring=h.get("grad_ring", 1),
            smoothness_lambda=h.get("smoothness_lambda", 0.05),
            residual_lambda=h.get("residual_lambda", 0.01),
            output_mode=h.get("output_mode", "delta"),
            device=device,
        )
        model.load_state_dict(ckpt["model_state"], strict=strict)
        model.to(device)
        return model, ckpt

    def extra_repr(self) -> str:
        return (
            f"nside={self.nside}, depth={len(self.feature_channels)}, "
            f"features={self.feature_channels}, in={self.in_channels}, out={self.out_channels}, "
            f"G={self.n_gauges}, gauge={self.gauge_type!r}, output_mode={self.output_mode!r}, "
            f"params={self.count_parameters():,}"
        )
