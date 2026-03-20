from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import time

import numpy as np
import torch

import healpix_geo


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]


@dataclass
class AlmCoeffs:
    """
    Container for local complex spherical harmonic coefficients.

    Parameters
    ----------
    alm : torch.Tensor
        Complex coefficient tensor with shape [K] or [B, K].
    l : torch.Tensor
        Degree indices with shape [K].
    m : torch.Tensor
        Order indices with shape [K]. Only m >= 0 is stored in V1.
    meta : dict
        Metadata associated with the transform.
    """

    alm: torch.Tensor
    l: torch.Tensor
    m: torch.Tensor
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.alm):
            raise TypeError("alm must be a torch.Tensor")
        if not torch.is_tensor(self.l):
            raise TypeError("l must be a torch.Tensor")
        if not torch.is_tensor(self.m):
            raise TypeError("m must be a torch.Tensor")
        if self.l.ndim != 1 or self.m.ndim != 1:
            raise ValueError("l and m must be 1D tensors")
        if self.l.shape != self.m.shape:
            raise ValueError("l and m must have the same shape")
        if self.alm.ndim not in (1, 2):
            raise ValueError("alm must have shape [K] or [B, K]")
        if self.alm.shape[-1] != self.l.numel():
            raise ValueError("The last alm dimension must match len(l) and len(m)")

    @property
    def n_modes(self) -> int:
        return int(self.l.numel())

    @property
    def is_batched(self) -> bool:
        return self.alm.ndim == 2

    def clone(self) -> "AlmCoeffs":
        return AlmCoeffs(
            alm=self.alm.clone(),
            l=self.l.clone(),
            m=self.m.clone(),
            meta=dict(self.meta),
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "AlmCoeffs":
        return AlmCoeffs(
            alm=self.alm.to(device=device, dtype=dtype),
            l=self.l.to(device=device),
            m=self.m.to(device=device),
            meta=dict(self.meta),
        )

    def cpu(self) -> "AlmCoeffs":
        return self.to(device="cpu")

    def cuda(self) -> "AlmCoeffs":
        return self.to(device="cuda")


class AlmTransform:
    """
    Local FFT-oriented spherical harmonic transform on a HEALPix patch.

    This implementation follows a two-stage local solver:

    1. local Fourier analysis on each observed iso-latitude ring,
    2. local Legendre analysis over the observed ring colatitudes.

    The design avoids explicit dense Y_lm evaluation over all pixels. It is
    intended as a practical starting point for large local patches and keeps
    only m >= 0 coefficients.

    Parameters
    ----------
    level : int
        HEALPix level such that nside = 2**level.
    cell_ids : array-like
        HEALPix cell identifiers at the given level.
    indexing_scheme : str, default="ring"
        The indexing scheme for the cell IDs.
    ellipsoid : {"sphere", "WGS84"}, default="sphere"
        Geometry model. Ignored in this version but stored for compatibility.
    method : {"fft", "alm"}, default="fft", Alm approximation, alm option under development.
        Analysis mode used in the Legendre stage.
    dtype : torch.dtype, default=torch.float32
        Real dtype for maps and geometry.
    device : str or torch.device, optional
        Execution device. If None, uses CUDA when available, else CPU.
    debug : bool, default=False
        Whether to store diagnostic timing information.

    Notes
    -----
    Current assumptions:
    - all input cells are at the same HEALPix level,
    - reconstruction is only performed on the same input cells,
    - input maps are real-valued.
    """

    def __init__(
        self,
        cell_ids: ArrayLike,
        level: int,
        indexing_scheme: str = "2D_array",
        ellipsoid: str = "sphere",
        method: str = "fft",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        debug: bool = False,
        lon: np.ndarray = None,
        lat: np.ndarray = None,
        weights: np.array = None
    ) -> None:

        self.level = int(level)
        self.indexing_scheme = str(indexing_scheme)
        self.ellipsoid = str(ellipsoid)
        self.method = str(method)
        self.dtype = dtype
        self.cdtype = self._infer_complex_dtype(dtype)
        self.device = self._resolve_device(device)
        self.debug = bool(debug)

        if self.ellipsoid not in {"sphere", "WGS84"}:
            raise ValueError("ellipsoid must be 'sphere' or 'WGS84'")
        if self.method not in {"fft", "alm"}:
            raise ValueError("method must be 'fft' or 'alm'")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.cell_ids = self._as_long_tensor(cell_ids, device=self.device)
        self.n_lat, self.n_lon = self.cell_ids.shape

        if self.indexing_scheme == "2D_array":
            self.lons_of_western_edge, _ = healpix_geo.ring.healpix_to_lonlat(cell_ids[:,0], level, ellipsoid=ellipsoid)
            self.lons_of_western_edge = self._as_real_tensor(self.lons_of_western_edge, device=self.device, dtype=self.dtype)

            #lon, lat = healpix_geo.ring.healpix_to_lonlat(self.cell_ids.flatten(), level, ellipsoid=ellipsoid)
            #self.idx_ordering = lon.argsort() # TODO: optimize knowing that it is already sorted
        else:
            raise NotImplementedError("For now, indexing_scheme must be '2D_array'")
        

        # TODO: make more robust computaion because of possible wrap-around at the 0/360 boundary, currently assumes no wrap-around and that lons are sorted in ascending order
        lon_range = healpix_geo.ring.healpix_to_lonlat(cell_ids[0,-1], level, ellipsoid=ellipsoid)[0]
        lon_range -= healpix_geo.ring.healpix_to_lonlat(cell_ids[0,0], level, ellipsoid=ellipsoid)[0]
        lon_range = lon_range[0]
        print("lon_range", lon_range)
        print("self.n_lon", self.n_lon)
        print("self.lons_of_western_edge - self.lons_of_western_edge.min()", self.lons_of_western_edge - self.lons_of_western_edge.min())
        pixel_shift = self.n_lon * (self.lons_of_western_edge - self.lons_of_western_edge.min()) / lon_range  # convert from longitude shift to pixel shift
        self.pixel_shift = pixel_shift
        #self.pixel_shift = torch.zeros_like(self.pixel_shift)  # TODO: remove after testing, currently set to zero for testing purposes
        #self.pixel_shift[1::2] = .5
        print("pixel_shift", self.pixel_shift)
        self.phase_shift = self.compute_phase_shift(-self.pixel_shift)

        # TODO: optimize knowing that it is already sorted
        #theta_uniq, idx_ring = np.unique(lat[self.idx_ordering], return_inverse=True)
    
        #self.n_rings = len(theta_uniq)
        #self.N_k = np.bincount(idx_ring)
        #self.N_max = int(np.max(self.N_k))
        #self.idx_ring = idx_ring
        #self.lon = (lon - np.min(lon))[self.idx_ordering]
        

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_phase_shift(self, x0, dim=-1):
        """
        Applique un décalage spatial x0 à un signal via sa FFT déjà calculée.

        Parameters
        ----------
        x0 : float or torch.Tensor
            Décalage à appliquer dans l'espace réel.
            Convention: f(x - x0) <-> F(k) * exp(-i 2π k x0)
        dim : int
            Dimension correspondant à la FFT 1D.

        Returns
        -------
        torch.Tensor
            Facteur de phase.
        """
        assert dim == -1

        # fréquences en cycles / unité de x
        f = torch.fft.fftfreq(self.n_lon, device=self.device, dtype=self.dtype)

        if True:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16,8))
            plt.imshow((f[None, :] * x0[:, None]).cpu()[:20, :], cmap='bwr')
            plt.colorbar(orientation='horizontal')
            plt.tight_layout()
            plt.show()

        # facteur de phase
        phase = torch.exp(2j * torch.pi * f[None, :] * x0[:, None])

        if False:
            print(phase)
            import matplotlib.pyplot as plt
            plt.imshow(phase.real.cpu())
            plt.colorbar()
            plt.show()

            plt.plot(phase.real.cpu()[:,180])
            plt.show()

        return phase
        
    def fft(
        self,
        data_: ArrayLike,
        pbc: bool = True,
    ) -> ArrayLike:
        data_ = self._as_real_tensor(data_, device=self.device, dtype=self.dtype)

        assert (self.pixel_shift >= 0).all() # assumes Eastern deviation only to be corrected 

        #data_shifted = torch.cat([data_[:,1:], data_[:,-1][:,None]], dim=1)  # shift data by one pixel to the right and pad by replicating the last pixel on the right
        data_shifted = torch.cat([data_[:,0][:,None], data_[:,:-1]], dim=1)  # shift data by one pixel to the right and pad by replicating the last pixel on the right

        data = (1-self.pixel_shift[:,None]) * data_ + self.pixel_shift[:,None] * data_shifted
        print("data input", data_)
        print("data corrected", data)

        # TODO: implement non PBC handling
        if pbc != True:
            raise NotImplementedError("Non-periodic boundary conditions are not yet implemented.")
        
        # 1D FFTs along longitudes, in parallel over latitude rings
        # TODO: implement rfft
        out_fft = torch.fft.fft(self._as_real_tensor(data, device=self.device, dtype=self.dtype), dim=-1)


        if True:
            import matplotlib.pyplot as plt
            plt.plot(out_fft[:,130].real.cpu())
            plt.plot(out_fft[:,130].imag.cpu())
            plt.show()

        # Apply phase shift to align with the original longitude
        #out_fft = out_fft * self.phase_shift
        # TODO: raise Warning if phase shift is large

        if True:
            import matplotlib.pyplot as plt
            plt.plot(out_fft[:,130].real.cpu())
            plt.plot(out_fft[:,130].imag.cpu())
            plt.show()

        if True:
            import matplotlib.pyplot as plt
            plt.imshow(np.fft.fftshift(np.abs(out_fft.detach().cpu())), norm='log')
            plt.colorbar(label='Power (log scale)', orientation='horizontal')
            plt.show()

        # 1D FFTs along latitudes, in parallel over longitudes
        # TODO: NUFFT or direct Legendre analysis for the second stage, currently only FFT stage implemented
        out_fft = torch.fft.fft(out_fft, dim=0) 

        return out_fft


    def ifft(
        self,
        data: ArrayLike,
    ) -> ArrayLike:
        """Reconstruct a map on the same local HEALPix cells."""
        o_fft = np.fft.ifft(data_fft)
    
        idata = np.zeros([self.size])
        for k in range(self.n_rings):
            idx = np.where(self.idx_ring==k)[0]
            inv_fft = o_fft[:,k]/self.weights[k]
            inv_fft = self.shift_from_fft_phase(inv_fft,-self.xa[idx[0]])
            idata[idx]=torch.fft.ifft(inv_fft)[0:self.N_k[k]].real
        return idata 

    
    @staticmethod
    def _infer_complex_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype == torch.float32:
            return torch.complex64
        if dtype == torch.float64:
            return torch.complex128
        raise ValueError("Unsupported real dtype")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    @staticmethod
    def _as_long_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    @staticmethod
    def _as_real_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)
