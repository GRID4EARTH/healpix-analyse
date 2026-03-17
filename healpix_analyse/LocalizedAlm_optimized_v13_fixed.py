#!/usr/bin/env python3
"""
LocalizedAlm_optimized_v13_fixed.py

CRITICAL FIX from v12: 
- PLM is now kept as REAL (float32) instead of being converted to complex
- Output dtype matches the original (float32 instead of float64)
- Uses element-wise multiply + sum instead of matmul (matches original gradient flow)

These changes ensure proper gradient flow through the real * complex multiplication,
which is essential for correct optimization dynamics in synthesis.

Strategy: Precompute all Legendre polynomials on CPU using numpy (which is
fast for sequential operations), store in CPU memory, and transfer to GPU
on-demand during anafast_localized.
"""

import numpy as np
import torch
import foscat.alm as alm_base

# =============================================================================
# Backend utilities
# =============================================================================
class _AlmBackendAdapter:
    def __init__(self, bk_backend):
        self.backend = bk_backend

def _as_alm_backend_object(obj):
    if hasattr(obj, "bk_log") and hasattr(obj, "bk_cast"):
        return _AlmBackendAdapter(obj)
    if hasattr(obj, "backend") and hasattr(obj.backend, "bk_log"):
        return obj
    raise TypeError("Invalid backend.")

def _get_torch_device_from_bk(bk):
    if hasattr(bk, "torch_device"): return bk.torch_device
    if hasattr(bk, "device"): return bk.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_or_compute_coefficients(lmax, device, dtype=None):
    """Dummy for compatibility."""
    pass


# =============================================================================
# Precompute PLM on CPU using numpy (fast)
# =============================================================================
def precompute_all_plm_numpy(co_th_np, lmax, nside, limit_range=1e10):
    """
    Precompute all P_lm on CPU using numpy.
    
    Returns list of numpy arrays (one per m value).
    This is MUCH faster than computing on GPU with Python loops.
    """
    n_rings = len(co_th_np)
    co_th = co_th_np.astype(np.float64)
    
    inv_limit = 1.0 / limit_range
    log_limit = np.log(limit_range)
    norm_base = np.sqrt(4.0 * np.pi) / (12.0 * nside * nside)
    
    plm_list = []
    
    print(f"  Precomputing PLM on CPU: lmax={lmax}, n_rings={n_rings}")
    
    for m in range(lmax + 1):
        if m % 500 == 0 and m > 0:
            print(f"    m={m}/{lmax}")
        
        n_l = lmax - m + 1
        result = np.zeros((n_l, n_rings), dtype=np.float64)
        ratio = np.zeros(n_l, dtype=np.float64)
        
        # P_mm
        if m == 0:
            result[0] = 1.0
        else:
            sin_th_sq = np.maximum(1.0 - co_th * co_th, 0.0)
            sign = 1.0 if (m % 2 == 0) else -1.0
            result[0] = sign * np.power(sin_th_sq, m / 2.0)
            
            df_log = sum(np.log(i) for i in range(2*m - 1, 0, -2)) if m > 0 else 0.0
            sum_log = np.sum(np.log(1 + np.arange(2*m))) if m > 0 else 0.0
            ratio[0] = df_log - 0.5 * sum_log
        
        if n_l > 1:
            # P_{m+1, m}
            result[1] = (2 * m + 1) * co_th * result[0]
            ratio[1] = ratio[0] - 0.5 * np.log(2 * m + 1) if m > 0 else 0.0
            
            max_val = np.max(np.abs(result[1]))
            if max_val > limit_range:
                result[0] *= inv_limit
                result[1] *= inv_limit
                ratio[0] += log_limit
                ratio[1] += log_limit
        
        # Recurrence
        for ell in range(m + 2, lmax + 1):
            i = ell - m
            A = (2 * ell - 1) / (ell - m)
            B = (ell + m - 1) / (ell - m)
            
            result[i] = A * co_th * result[i-1] - B * result[i-2]
            ratio[i] = ratio[i-1] + 0.5 * np.log(ell - m) - 0.5 * np.log(ell + m)
            
            max_val = np.max(np.abs(result[i]))
            if max_val > limit_range:
                result[i-1] *= inv_limit
                result[i] *= inv_limit
                ratio[i-1] += log_limit
                ratio[i] += log_limit
        
        # Apply normalization
        exp_ratio = np.exp(ratio).reshape(-1, 1)
        plm_list.append((result * exp_ratio * norm_base).astype(np.float32))  # float32 to save memory
    
    print(f"  PLM precomputation complete.")
    
    # Estimate memory usage
    total_bytes = sum(p.nbytes for p in plm_list)
    print(f"  PLM CPU memory: {total_bytes / 1e9:.2f} GB")
    
    return plm_list


# =============================================================================
# LocalizedAlm class
# =============================================================================
class LocalizedAlm(alm_base.alm):
    """
    Localized spherical harmonic analysis with CPU-precomputed PLM.
    
    PLM is precomputed on CPU (fast numpy), stored in CPU RAM,
    and transferred to GPU on-demand.
    """
    
    def __init__(self, nside, patch_idx_ring, backend, lmax=None, lmax_compute=None):
        patch_idx_ring = np.asarray(patch_idx_ring, dtype=np.int64)
        if patch_idx_ring.size > 1:
            patch_idx_ring = np.sort(patch_idx_ring)
        
        alm_backend = _as_alm_backend_object(backend)
        super().__init__(backend=alm_backend, lmax=lmax, nside=nside)
        
        self.nside = nside
        self.patch_idx_ring = patch_idx_ring
        self.device = _get_torch_device_from_bk(self.backend)
        self.dtype = self.backend.all_bk_type
        self.lmax_full = int(self.lmax)
        self.lmax_compute = lmax_compute if lmax_compute is not None else self.lmax_full
        
        self._precompute_geometry(nside, patch_idx_ring)
        
        self.shift_ph(nside)
        full_shift = self.matrix_shift_ph[nside]
        idx_t = torch.as_tensor(self.active_ring_indices_global, dtype=torch.long, device=full_shift.device)
        self.active_shift_ph = full_shift.index_select(0, idx_t)
        
        self._build_torch_scatter_tables()
        
        # Get cos(theta) for active rings
        full_theta = self.ring_th(nside)
        active_theta = full_theta[self.active_ring_indices_global]
        co_th_np = np.cos(active_theta)
        
        # Precompute all PLM on CPU (fast numpy)
        self.plm_list_cpu = precompute_all_plm_numpy(
            co_th_np, self.lmax_compute, nside
        )
        
        print(f"  LocalizedAlm ready: {len(self.active_rings)} rings, lmax_compute={self.lmax_compute}")

    def _precompute_geometry(self, nside, patch_idx):
        rings_info = []
        pix_cursor = 0
        
        for k in range(nside - 1):
            length = 4 * (k + 1)
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        for k in range(2 * nside + 1):
            length = 4 * nside
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        for k in range(nside - 1):
            length = 4 * (nside - 1 - k)
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        active_rings = []
        active_ring_indices_global = []
        u_indices = []
        ring_batch_indices = []
        ring_pix_indices = []
        
        for r in rings_info:
            s = r["start"]
            e = s + r["length"]
            start_search = np.searchsorted(patch_idx, s)
            end_search = np.searchsorted(patch_idx, e)
            
            if end_search > start_search:
                n_active_pixels = end_search - start_search
                pixels_global = patch_idx[start_search:end_search]
                pixels_local_ring = pixels_global - s
                
                active_rings.append(r)
                active_ring_indices_global.append(r["global_idx"])
                u_indices.append(np.arange(start_search, end_search, dtype=np.int64))
                local_ring_idx = len(active_rings) - 1
                ring_batch_indices.append(np.full(n_active_pixels, local_ring_idx, dtype=np.int64))
                ring_pix_indices.append(pixels_local_ring.astype(np.int64))
        
        self.active_rings = active_rings
        self.active_ring_indices_global = np.asarray(active_ring_indices_global, dtype=np.int64)
        
        if len(u_indices) > 0:
            self.scatter_src = np.concatenate(u_indices)
            self.scatter_dst_ring = np.concatenate(ring_batch_indices)
            self.scatter_dst_pix = np.concatenate(ring_pix_indices)
        else:
            self.scatter_src = np.array([], dtype=np.int64)
            self.scatter_dst_ring = np.array([], dtype=np.int64)
            self.scatter_dst_pix = np.array([], dtype=np.int64)

    def _build_torch_scatter_tables(self):
        self.group_info = {}
        if self.scatter_src.size == 0:
            return
        
        t_src = torch.as_tensor(self.scatter_src, dtype=torch.long, device=self.device)
        t_dst_ring = torch.as_tensor(self.scatter_dst_ring, dtype=torch.long, device=self.device)
        t_dst_pix = torch.as_tensor(self.scatter_dst_pix, dtype=torch.long, device=self.device)
        
        rings_by_length = {}
        for i, r in enumerate(self.active_rings):
            rings_by_length.setdefault(int(r["length"]), []).append(i)
        
        n_active = len(self.active_rings)
        for L, local_rings in rings_by_length.items():
            local_rings_t = torch.as_tensor(local_rings, dtype=torch.long, device=self.device)
            ring_to_pos = torch.full((n_active,), -1, dtype=torch.long, device=self.device)
            ring_to_pos[local_rings_t] = torch.arange(local_rings_t.numel(), device=self.device, dtype=torch.long)
            sel = ring_to_pos[t_dst_ring] >= 0
            sel_idx = torch.nonzero(sel, as_tuple=False).flatten()
            if sel_idx.numel() == 0:
                continue
            self.group_info[int(L)] = {
                "L": int(L),
                "local_rings": local_rings_t,
                "src": t_src[sel_idx],
                "dst_ring": ring_to_pos[t_dst_ring[sel_idx]],
                "dst_pix": t_dst_pix[sel_idx],
                "n_group": int(local_rings_t.numel())
            }

    def comp_tf_localized(self, u_patch):
        """Compute FFT on active rings."""
        if not isinstance(u_patch, torch.Tensor):
            u_patch = self.backend.bk_cast(u_patch)
        else:
            if u_patch.dtype != self.dtype:
                u_patch = u_patch.to(dtype=self.dtype)
            u_patch = u_patch.to(self.device)
        
        if u_patch.dim() == 1:
            u_patch = u_patch.unsqueeze(0)
        
        n_batch = u_patch.shape[0]
        n_active = len(self.active_rings)
        target_dim = 3 * self.nside
        
        ft_im = torch.zeros(
            (n_batch, n_active, target_dim),
            dtype=self.backend.all_cbk_type,
            device=self.device
        )
        
        for L, info in self.group_info.items():
            dense = torch.zeros((n_batch, info["n_group"], L), dtype=self.dtype, device=self.device)
            vals = u_patch.index_select(1, info["src"])
            base = (torch.arange(n_batch, device=self.device, dtype=torch.long) * (info["n_group"] * L)).unsqueeze(1)
            flat_idx = (base + info["dst_ring"].unsqueeze(0) * L + info["dst_pix"].unsqueeze(0)).reshape(-1)
            dense.view(-1).scatter_(0, flat_idx, vals.reshape(-1))
            
            r = torch.fft.rfft(dense, dim=-1)
            full_fft = torch.cat([r, torch.conj(torch.flip(r[..., 1:-1], dims=[-1]))], dim=-1)
            if full_fft.shape[-1] < target_dim + 1:
                full_fft = full_fft.repeat(1, 1, target_dim // full_fft.shape[-1] + 1)
            ft_im.index_copy_(1, info["local_rings"], full_fft[..., :target_dim])
        
        return ft_im * self.active_shift_ph.unsqueeze(0)

    def anafast_localized(self, u_patch, map2_patch=None, bandpowers_binning=None):
        """
        Compute pseudo-Cl using CPU-precomputed PLM.
        
        Transfers PLM to GPU on-demand, uses it, then deletes.
        
        IMPORTANT: PLM is kept as REAL (float32), matching the original
        LocalizedAlm.py behavior. This ensures proper gradient flow through
        the real * complex multiplication.
        """
        lmax = self.lmax_compute
        
        # Compute FFT once
        ft_im = self.comp_tf_localized(u_patch)
        ft_im2 = self.comp_tf_localized(map2_patch) if map2_patch is not None else ft_im
        
        # Setup output - use same dtype as original (backend's real type)
        out_dtype = self.dtype  # float32, matching original
        
        if bandpowers_binning is not None:
            if not isinstance(bandpowers_binning, torch.Tensor):
                bandpowers_binning = torch.as_tensor(
                    bandpowers_binning, device=self.device, dtype=out_dtype
                )
            else:
                bandpowers_binning = bandpowers_binning.to(self.device, dtype=out_dtype)
            n_bins = bandpowers_binning.shape[0]
            cl_out = torch.zeros(u_patch.shape[0], n_bins, dtype=out_dtype, device=self.device)
        else:
            cl_out = torch.zeros(u_patch.shape[0], lmax + 1, dtype=out_dtype, device=self.device)
        
        # Process each m value
        for m in range(lmax + 1):
            # Transfer PLM from CPU to GPU as REAL tensor (NOT complex!)
            # This matches the original LocalizedAlm.py: plm = self.backend.bk_cast(plm_np)
            plm_np = self.plm_list_cpu[m]  # [n_l, n_rings], float32
            plm = torch.as_tensor(plm_np, device=self.device, dtype=out_dtype)  # REAL, float32
            
            ft_slice = ft_im[:, :, m]    # [B, N_active], complex
            ft_slice2 = ft_im2[:, :, m]  # [B, N_active], complex
            
            # Compute alm using element-wise multiplication + sum (like original)
            # Original: alm_val = torch.sum(plm.unsqueeze(0) * ft_slice.unsqueeze(1), dim=2)
            # plm: [n_l, n_rings] -> [1, n_l, n_rings]
            # ft_slice: [B, n_rings] -> [B, 1, n_rings]
            # multiply: [B, n_l, n_rings]
            # sum over rings: [B, n_l]
            alm_val = torch.sum(plm.unsqueeze(0) * ft_slice.unsqueeze(1), dim=2)
            alm_val2 = torch.sum(plm.unsqueeze(0) * ft_slice2.unsqueeze(1), dim=2)
            
            # |alm|^2
            tmp = torch.real(alm_val * torch.conj(alm_val2))
            
            if m > 0:
                tmp = 2.0 * tmp
            
            # Accumulate (same dtype as output)
            if bandpowers_binning is not None:
                matrix_slice = bandpowers_binning[:, m:lmax + 1]
                cl_out = cl_out + torch.matmul(tmp, matrix_slice.T)
            else:
                cl_out[:, m:lmax + 1] = cl_out[:, m:lmax + 1] + tmp
            
            # Free GPU memory
            del plm, alm_val, alm_val2, tmp
        
        return cl_out


# =============================================================================
# Test
# =============================================================================
def test_localized_anafast():
    import healpy as hp
    import time
    
    print("\n" + "="*70)
    print("Testing LocalizedAlm_optimized_v13_fixed")
    print("="*70)
    
    nside_test = 64
    npix = 12 * nside_test**2
    lmax = 3 * nside_test - 1
    
    np.random.seed(42)
    full_map = np.random.randn(npix).astype(np.float32)
    
    vec = hp.ang2vec(np.pi / 2.5, np.pi / 3.0)
    radius = np.radians(20.0)
    patch_idx = hp.query_disc(nside_test, vec, radius, nest=False)
    patch_idx = np.sort(patch_idx).astype(np.int64)
    
    mask = np.zeros(npix, dtype=np.float32)
    mask[patch_idx] = 1.0
    masked_map = full_map * mask
    
    from foscat.BkTorch import BkTorch
    bk = BkTorch(all_type="float32", silent=True)
    
    alm_std = alm_base.alm(nside=nside_test, backend=_AlmBackendAdapter(bk), lmax=lmax)
    t_map = bk.bk_cast(masked_map[None, :])
    cl_ref, _ = alm_std.anafast(t_map, nest=False, spin=0, axes=1)
    cl_ref = cl_ref.detach().cpu().numpy()[0]
    
    patch_data = full_map[patch_idx]
    t_patch = bk.bk_cast(patch_data[None, :])
    
    print("\nInitializing LocalizedAlm (includes CPU precomputation)...")
    t0 = time.time()
    alm_loc = LocalizedAlm(nside_test, patch_idx, backend=bk, lmax=lmax)
    t_init = time.time() - t0
    print(f"Initialization time: {t_init:.2f}s")
    
    print("\nRunning anafast_localized...")
    t0 = time.time()
    cl_loc = alm_loc.anafast_localized(t_patch)
    t_first = time.time() - t0
    print(f"First call: {t_first:.3f}s")
    
    cl_loc = cl_loc.detach().cpu().numpy()[0]
    
    lmin = 2
    diff = np.abs(cl_ref[lmin:] - cl_loc[lmin:])
    rel_diff = diff / (np.abs(cl_ref[lmin:]) + 1e-30)
    
    print(f"\nResults:")
    print(f"  Max Relative Error: {np.max(rel_diff):.6e}")
    print(f"  Mean Relative Error: {np.mean(rel_diff):.6e}")
    
    print(f"\nSpectrum comparison:")
    for ell in [2, 50, 100, 150, 190]:
        ratio = cl_loc[ell] / (cl_ref[ell] + 1e-30)
        print(f"  ell={ell:3d}: ref={cl_ref[ell]:.4e}, loc={cl_loc[ell]:.4e}, ratio={ratio:.6f}")
    
    # Speed test
    print(f"\nSpeed test (10 calls)...")
    t0 = time.time()
    for _ in range(10):
        _ = alm_loc.anafast_localized(t_patch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_avg = (time.time() - t0) / 10
    print(f"Average time per call: {t_avg:.3f}s")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    
    if np.max(rel_diff) < 1e-4:
        print("\n>>> SUCCESS!")
        return True
    else:
        print("\n>>> FAILURE")
        return False


if __name__ == "__main__":
    test_localized_anafast()
