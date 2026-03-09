from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import math
import time

import numpy as np
import torch

try:
    import healpy as hp
except ImportError:  # pragma: no cover
    hp = None


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]


@dataclass
class AlmCoeffs:
    """
    Container for local complex spherical harmonic coefficients.

    Parameters
    ----------
    alm : torch.Tensor
        Complex coefficient tensor with shape [M] or [B, M].
    l : torch.Tensor
        Degree indices with shape [M].
    m : torch.Tensor
        Order indices with shape [M]. In V1, only m >= 0 is stored.
    meta : dict
        Arbitrary metadata associated with the coefficients.

    Notes
    -----
    This container is intentionally lightweight. It stores the coefficient
    vector and the corresponding spectral indices, but it does not own the
    transform geometry. Reconstruction must be done with a compatible
    `AlmTransform` instance.
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
            raise ValueError("alm must have shape [M] or [B, M]")

        if self.alm.shape[-1] != self.l.numel():
            raise ValueError("Last dimension of alm must match the number of modes")

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the coefficient tensor."""
        return self.alm.shape

    @property
    def batch_shape(self) -> torch.Size:
        """Return the batch shape, empty for unbatched coefficients."""
        return self.alm.shape[:-1]

    @property
    def n_modes(self) -> int:
        """Return the number of stored spectral modes."""
        return int(self.l.numel())

    @property
    def is_batched(self) -> bool:
        """Return True if coefficients have a batch dimension."""
        return self.alm.ndim == 2

    @property
    def device(self) -> torch.device:
        """Return the storage device."""
        return self.alm.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the complex dtype of the coefficient tensor."""
        return self.alm.dtype

    def clone(self) -> "AlmCoeffs":
        """Return a deep copy of the coefficient container."""
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
        """
        Move coefficients to another device and/or dtype.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device.
        dtype : torch.dtype, optional
            Target complex dtype.

        Returns
        -------
        AlmCoeffs
            New coefficient container.
        """
        return AlmCoeffs(
            alm=self.alm.to(device=device, dtype=dtype),
            l=self.l.to(device=device),
            m=self.m.to(device=device),
            meta=dict(self.meta),
        )

    def cpu(self) -> "AlmCoeffs":
        """Return a CPU copy."""
        return self.to(device="cpu")

    def cuda(self) -> "AlmCoeffs":
        """Return a CUDA copy."""
        return self.to(device="cuda")


class AlmTransform:
    """
    Local truncated spherical harmonic transform on a HEALPix patch.

    This class defines a local spherical harmonic analysis / synthesis operator
    on a fixed subset of HEALPix cells at a single level. The transform is not
    a full-sphere global HEALPix alm transform. Instead, it works on a local
    patch and selects a subset of spectral modes according to a window rule.

    Parameters
    ----------
    level : int
        HEALPix level such that nside = 2**level.
    cell_ids : array-like
        Nested HEALPix cell identifiers at the given level.
    ellipsoid : {"sphere", "WGS84"}, default="sphere"
        Geometry model. Ignored in V1 but stored for future compatibility.
    lmax_compute : int, optional
        Maximum harmonic degree effectively used in the local transform.
        If None, defaults to 3 * nside.
    mode : {"window"}, default="window"
        Spectral mode selection strategy.
    method : {"lstsq", "project"}, default="lstsq"
        Default forward transform method.
    lambda_reg : float, default=1e-6
        Tikhonov regularization strength for `method="lstsq"`.
    dtype : torch.dtype, default=torch.float32
        Real dtype used for maps and geometry.
    device : str or torch.device, optional
        Execution device. If None, uses CUDA when available, else CPU.
    chunk_size_pixels : int, default=8192
        Number of pixels processed per block.
    chunk_size_modes : int, default=2048
        Number of modes processed per block.
    precompute : {"none", "angles", "auto", "design"}, default="auto"
        Precomputation level.
    cache_design : bool, default=False
        Whether to cache harmonic design blocks when memory allows it.
    nest : bool, default=True
        Ordering convention. Only nested ordering is supported in V1.
    max_modes : int, optional
        Optional safety cap on the number of selected modes.
    debug : bool, default=False
        Whether to store diagnostic timing and geometry information.

    Notes
    -----
    V1 design choices:
    - input cells are all at the same level
    - input ordering is nested
    - the transform reconstructs only on the same input cells
    - only m >= 0 coefficients are stored
    - spherical harmonic coefficients are complex
    - map input is assumed real-valued
    """

    def __init__(
        self,
        level: int,
        cell_ids: ArrayLike,
        ellipsoid: str = "sphere",
        lmax_compute: Optional[int] = None,
        mode: str = "window",
        method: str = "lstsq",
        lambda_reg: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size_pixels: int = 8192,
        chunk_size_modes: int = 2048,
        precompute: str = "auto",
        cache_design: bool = False,
        nest: bool = True,
        max_modes: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        self.level = int(level)
        self.nside = 2 ** self.level
        self.ellipsoid = str(ellipsoid)
        self.mode = str(mode)
        self.method = str(method)
        self.lambda_reg = float(lambda_reg)
        self.dtype = dtype
        self.cdtype = self._infer_complex_dtype(dtype)
        self.device = self._resolve_device(device)
        self.chunk_size_pixels = int(chunk_size_pixels)
        self.chunk_size_modes = int(chunk_size_modes)
        self.precompute = str(precompute)
        self.cache_design = bool(cache_design)
        self.nest = bool(nest)
        self.max_modes = max_modes
        self.debug = bool(debug)

        if self.ellipsoid not in {"sphere", "WGS84"}:
            raise ValueError("ellipsoid must be 'sphere' or 'WGS84'")

        if self.mode != "window":
            raise ValueError("Only mode='window' is supported in V1")

        if self.method not in {"lstsq", "project"}:
            raise ValueError("method must be 'lstsq' or 'project'")

        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")

        if not self.nest:
            raise ValueError("Only nested input ordering is supported in V1")

        if hp is None:
            raise ImportError(
                "healpy is required for the current V1 geometry backend. "
                "Install healpy or replace the geometry backend with healpix_geo."
            )

        self.cell_ids = self._as_long_tensor(cell_ids, device=self.device)
        if self.cell_ids.ndim != 1:
            raise ValueError("cell_ids must be a 1D array-like")
        if self.cell_ids.numel() == 0:
            raise ValueError("cell_ids must not be empty")

        self.n_pixels = int(self.cell_ids.numel())
        self.npix_full = 12 * self.nside * self.nside
        self.lmax_theoretical = 3 * self.nside
        self.lmax_compute = (
            int(lmax_compute) if lmax_compute is not None else self.lmax_theoretical
        )
        self.pixel_area = 4.0 * math.pi / float(self.npix_full)

        self._debug_info: Dict[str, Any] = {}
        self._design_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

        # Window rule parameters for V1.
        self.window_alpha = 1.5
        self.window_margin_m = 8
        self.window_min_m = 2

        # Geometry
        t0 = time.perf_counter()
        self.theta, self.phi = self._compute_angles_from_nested_cell_ids(self.cell_ids)
        self.cos_theta = torch.cos(self.theta)
        self.sin_theta = torch.sin(self.theta)
        self._patch_center_vec = self._compute_patch_center_vector()
        self.patch_theta0, self.patch_phi0 = self._vector_to_angles(self._patch_center_vec)
        self.patch_radius = self._compute_patch_radius()
        if self.debug:
            self._debug_info["geometry_time_sec"] = time.perf_counter() - t0

        # Spectral index selection
        t0 = time.perf_counter()
        self.l, self.m = self._build_mode_index()
        self.n_modes = int(self.l.numel())
        if self.debug:
            self._debug_info["mode_build_time_sec"] = time.perf_counter() - t0

        if self.n_modes == 0:
            raise RuntimeError("Window mode selection produced zero modes")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def from_map(
        self,
        data: ArrayLike,
        method: Optional[str] = None,
        lambda_reg: Optional[float] = None,
    ) -> AlmCoeffs:
        """
        Compute local complex spherical harmonic coefficients from a map.

        Parameters
        ----------
        data : array-like
            Input map values on the transform cells. Supported shapes are [N]
            and [B, N].
        method : {"lstsq", "project"}, optional
            Override the default forward method.
        lambda_reg : float, optional
            Override the default Tikhonov regularization parameter.

        Returns
        -------
        AlmCoeffs
            Local complex coefficients with associated (l, m) indices.
        """
        method = self.method if method is None else str(method)
        lambda_reg = self.lambda_reg if lambda_reg is None else float(lambda_reg)

        if method not in {"lstsq", "project"}:
            raise ValueError("method must be 'lstsq' or 'project'")

        x, squeezed = self._prepare_map_input(data)

        if method == "project":
            alm = self._from_map_project(x)
        else:
            alm = self._from_map_lstsq(x, lambda_reg=lambda_reg)

        if squeezed:
            alm = alm.squeeze(0)

        meta = {
            "level": self.level,
            "nside": self.nside,
            "n_pixels": self.n_pixels,
            "lmax_compute": self.lmax_compute,
            "mode": self.mode,
            "method": method,
            "lambda_reg": lambda_reg,
            "ellipsoid": self.ellipsoid,
            "nested": self.nest,
        }
        return AlmCoeffs(alm=alm, l=self.l, m=self.m, meta=meta)

    def to_map(
        self,
        coeffs: Union[AlmCoeffs, torch.Tensor],
    ) -> torch.Tensor:
        """
        Reconstruct a map on the transform cells from local coefficients.

        Parameters
        ----------
        coeffs : AlmCoeffs or torch.Tensor
            Coefficient container or raw coefficient tensor with shape [M]
            or [B, M].

        Returns
        -------
        torch.Tensor
            Reconstructed real-valued map with shape [N] or [B, N].
        """
        alm, squeezed = self._prepare_coeff_input(coeffs)
        out = self._synthesis_map(alm)
        out = out.real

        if squeezed:
            out = out.squeeze(0)
        return out

    def smooth(
        self,
        data: ArrayLike,
        sigma: Optional[float] = None,
        fwhm: Optional[float] = None,
        method: Optional[str] = None,
        lambda_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply a radial Gaussian smoothing in local harmonic space.

        Parameters
        ----------
        data : array-like
            Input map values on the transform cells, shape [N] or [B, N].
        sigma : float, optional
            Gaussian sigma in radians.
        fwhm : float, optional
            Gaussian FWHM in radians.
        method : {"lstsq", "project"}, optional
            Forward transform method override.
        lambda_reg : float, optional
            Tikhonov regularization override.

        Returns
        -------
        torch.Tensor
            Smoothed map on the same cells.
        """
        coeffs = self.from_map(data, method=method, lambda_reg=lambda_reg)
        coeffs_smoothed = self.filter_alm(coeffs, sigma=sigma, fwhm=fwhm)
        return self.to_map(coeffs_smoothed)

    def filter_alm(
        self,
        coeffs: AlmCoeffs,
        beam: Optional[ArrayLike] = None,
        sigma: Optional[float] = None,
        fwhm: Optional[float] = None,
    ) -> AlmCoeffs:
        """
        Apply a radial filter in harmonic space.

        Parameters
        ----------
        coeffs : AlmCoeffs
            Input coefficient container.
        beam : array-like, optional
            Radial filter b_l with shape [lmax_filter + 1].
        sigma : float, optional
            Gaussian sigma in radians.
        fwhm : float, optional
            Gaussian FWHM in radians.

        Returns
        -------
        AlmCoeffs
            Filtered coefficients.
        """
        if beam is not None and (sigma is not None or fwhm is not None):
            raise ValueError("Specify either beam or sigma/fwhm, not both")

        if beam is None:
            sigma_val = self._resolve_sigma(sigma=sigma, fwhm=fwhm)
            beam_t = self._gaussian_beam(self.l, sigma=sigma_val)
        else:
            beam_t = self._as_real_tensor(beam, device=coeffs.alm.device, dtype=self.dtype)
            lmax_beam = beam_t.numel() - 1
            if int(coeffs.l.max().item()) > lmax_beam:
                raise ValueError("Provided beam does not cover all requested l values")
            beam_t = beam_t[coeffs.l]

        while beam_t.ndim < coeffs.alm.ndim:
            beam_t = beam_t.unsqueeze(0)

        alm_filtered = coeffs.alm * beam_t.to(dtype=coeffs.alm.real.dtype)
        return AlmCoeffs(
            alm=alm_filtered,
            l=coeffs.l,
            m=coeffs.m,
            meta=dict(coeffs.meta),
        )

    def get_window_modes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the selected local spectral indices.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Degree and order tensors, both with shape [M].
        """
        return self.l.clone(), self.m.clone()

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the transform geometry and spectral configuration.
        """
        out = {
            "level": self.level,
            "nside": self.nside,
            "n_pixels": self.n_pixels,
            "npix_full": self.npix_full,
            "pixel_area": self.pixel_area,
            "lmax_theoretical": self.lmax_theoretical,
            "lmax_compute": self.lmax_compute,
            "n_modes": self.n_modes,
            "mode": self.mode,
            "method": self.method,
            "lambda_reg": self.lambda_reg,
            "dtype": str(self.dtype),
            "cdtype": str(self.cdtype),
            "device": str(self.device),
            "patch_theta0": float(self.patch_theta0.item()),
            "patch_phi0": float(self.patch_phi0.item()),
            "patch_radius": float(self.patch_radius.item()),
            "nested": self.nest,
            "ellipsoid": self.ellipsoid,
        }
        if self.debug:
            out["debug"] = dict(self._debug_info)
        return out

    # ---------------------------------------------------------------------
    # Forward operators
    # ---------------------------------------------------------------------

    def _from_map_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Direct weighted projection:
            a = Y^H W f

        Parameters
        ----------
        x : torch.Tensor
            Real map tensor with shape [B, N].

        Returns
        -------
        torch.Tensor
            Complex coefficients with shape [B, M].
        """
        bsz = x.shape[0]
        out = torch.zeros(
            (bsz, self.n_modes),
            dtype=self.cdtype,
            device=self.device,
        )

        weight = torch.as_tensor(
            self.pixel_area,
            dtype=self.dtype,
            device=self.device,
        )

        for p0 in range(0, self.n_pixels, self.chunk_size_pixels):
            p1 = min(p0 + self.chunk_size_pixels, self.n_pixels)
            y_block = self._eval_Y_pixels_all_modes(p0, p1)  # [P, M]
            x_block = x[:, p0:p1]  # [B, P]

            # out += conj(Y)^T @ (W * x)
            out = out + torch.einsum(
                "pm,bp->bm",
                torch.conj(y_block),
                x_block * weight,
            )

        return out

    def _from_map_lstsq(self, x: torch.Tensor, lambda_reg: float) -> torch.Tensor:
        """
        Regularized local least-squares solve:
            (Y^H W Y + lambda I) a = Y^H W f

        Parameters
        ----------
        x : torch.Tensor
            Real map tensor with shape [B, N].
        lambda_reg : float
            Tikhonov regularization parameter.

        Returns
        -------
        torch.Tensor
            Complex coefficients with shape [B, M].
        """
        t0 = time.perf_counter()

        G = self._build_gram_matrix()  # [M, M]
        rhs = self._build_rhs(x)       # [B, M]

        eye = torch.eye(self.n_modes, dtype=self.cdtype, device=self.device)
        A = G + lambda_reg * eye

        # Solve one system per batch element.
        # We solve A^T? No, standard solve with RHS in columns:
        # A @ X = RHS^T
        sol = torch.linalg.solve(A, rhs.T).T

        if self.debug:
            self._debug_info["lstsq_total_time_sec"] = time.perf_counter() - t0
        return sol

    def _build_gram_matrix(self) -> torch.Tensor:
        """
        Build the regular LS Gram matrix G = Y^H W Y.

        Returns
        -------
        torch.Tensor
            Complex Hermitian matrix with shape [M, M].

        Notes
        -----
        This V1 implementation forms the full Gram matrix explicitly.
        For very large M, this will become the main memory and time bottleneck.
        A future version may use:
        - low-rank approximations,
        - iterative solvers,
        - blockwise normal operator application,
        - localized basis compression.
        """
        t0 = time.perf_counter()
        weight = torch.as_tensor(
            self.pixel_area,
            dtype=self.dtype,
            device=self.device,
        )

        G = torch.zeros(
            (self.n_modes, self.n_modes),
            dtype=self.cdtype,
            device=self.device,
        )

        for p0 in range(0, self.n_pixels, self.chunk_size_pixels):
            p1 = min(p0 + self.chunk_size_pixels, self.n_pixels)
            y_block = self._eval_Y_pixels_all_modes(p0, p1)  # [P, M]
            G = G + torch.einsum("pm,pn->mn", torch.conj(y_block), y_block * weight)

        if self.debug:
            self._debug_info["gram_time_sec"] = time.perf_counter() - t0
        return G

    def _build_rhs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build the right-hand side b = Y^H W f.

        Parameters
        ----------
        x : torch.Tensor
            Real map tensor with shape [B, N].

        Returns
        -------
        torch.Tensor
            Complex tensor with shape [B, M].
        """
        t0 = time.perf_counter()
        bsz = x.shape[0]
        rhs = torch.zeros((bsz, self.n_modes), dtype=self.cdtype, device=self.device)
        weight = torch.as_tensor(
            self.pixel_area,
            dtype=self.dtype,
            device=self.device,
        )

        for p0 in range(0, self.n_pixels, self.chunk_size_pixels):
            p1 = min(p0 + self.chunk_size_pixels, self.n_pixels)
            y_block = self._eval_Y_pixels_all_modes(p0, p1)  # [P, M]
            x_block = x[:, p0:p1]
            rhs = rhs + torch.einsum(
                "pm,bp->bm",
                torch.conj(y_block),
                x_block * weight,
            )

        if self.debug:
            self._debug_info["rhs_time_sec"] = time.perf_counter() - t0
        return rhs

    # ---------------------------------------------------------------------
    # Synthesis
    # ---------------------------------------------------------------------

    def _synthesis_map(self, alm: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct map values from local coefficients.

        Parameters
        ----------
        alm : torch.Tensor
            Complex coefficients with shape [B, M].

        Returns
        -------
        torch.Tensor
            Complex reconstructed map with shape [B, N].

        Notes
        -----
        Because only m >= 0 is stored, reconstruction uses the standard
        real-map symmetry:
            f(theta, phi) = Re[a_{l0} Y_{l0}] + 2 Re[sum_{m>0} a_{lm} Y_{lm}]
        under the assumption of a real-valued map.
        """
        bsz = alm.shape[0]
        out = torch.zeros((bsz, self.n_pixels), dtype=self.cdtype, device=self.device)

        m_is_zero = self.m == 0
        m_is_pos = self.m > 0

        for p0 in range(0, self.n_pixels, self.chunk_size_pixels):
            p1 = min(p0 + self.chunk_size_pixels, self.n_pixels)
            y_block = self._eval_Y_pixels_all_modes(p0, p1)  # [P, M]

            block = torch.zeros((bsz, p1 - p0), dtype=self.cdtype, device=self.device)

            if torch.any(m_is_zero):
                y0 = y_block[:, m_is_zero]
                a0 = alm[:, m_is_zero]
                block = block + torch.einsum("pm,bm->bp", y0, a0)

            if torch.any(m_is_pos):
                yp = y_block[:, m_is_pos]
                ap = alm[:, m_is_pos]
                block = block + 2.0 * torch.einsum("pm,bm->bp", yp, ap).real

            out[:, p0:p1] = block

        return out

    # ---------------------------------------------------------------------
    # Harmonic basis evaluation
    # ---------------------------------------------------------------------

    def _eval_Y_pixels_all_modes(self, p0: int, p1: int) -> torch.Tensor:
        """
        Evaluate the local harmonic design block Y on a pixel slice.

        Parameters
        ----------
        p0, p1 : int
            Pixel interval [p0, p1).

        Returns
        -------
        torch.Tensor
            Complex design block with shape [P, M].

        Notes
        -----
        This method is the critical numerical kernel of the class.

        The current skeleton intentionally leaves room for a proper torch-native
        implementation based on associated Legendre functions and exp(i m phi).
        For V1 prototyping, it can be backed by:
        - scipy.special.sph_harm for validation,
        - healpy helpers when appropriate,
        - or a custom torch recurrence.

        The returned block must follow the same harmonic normalization used
        consistently in both analysis and synthesis.
        """
        cache_key = (p0, p1, 0, self.n_modes)
        if self.cache_design and cache_key in self._design_cache:
            return self._design_cache[cache_key]

        theta = self.theta[p0:p1]
        phi = self.phi[p0:p1]

        # -----------------------------------------------------------------
        # IMPORTANT:
        # Replace this placeholder with a proper torch-native implementation.
        #
        # Expected output:
        #   Y[p, k] = Y_{l_k}^{m_k}(theta_p, phi_p)
        #
        # Constraints:
        # - output shape: [P, M]
        # - dtype: self.cdtype
        # - device: self.device
        # - convention must match analysis/synthesis
        # -----------------------------------------------------------------
        Y = self._eval_spherical_harmonics_placeholder(theta, phi, self.l, self.m)

        if self.cache_design:
            self._design_cache[cache_key] = Y
        return Y

    def _eval_spherical_harmonics_placeholder(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-native complex spherical harmonics evaluator for m >= 0.

        Parameters
        ----------
        theta : torch.Tensor
            Colatitudes in radians, shape [P].
        phi : torch.Tensor
            Longitudes in radians, shape [P].
        l : torch.Tensor
            Harmonic degrees, shape [M].
        m : torch.Tensor
            Harmonic orders, shape [M], with m >= 0.

        Returns
        -------
        torch.Tensor
            Complex spherical harmonics Y_lm(theta, phi), shape [P, M].

        Notes
        -----
        Conventions:
        - complex spherical harmonics
        - Condon-Shortley phase included
        - orthonormal normalization on the sphere
        - only m >= 0 is stored
        """
        if theta.ndim != 1 or phi.ndim != 1:
            raise ValueError("theta and phi must be 1D tensors")
        if theta.shape != phi.shape:
            raise ValueError("theta and phi must have the same shape")
        if l.ndim != 1 or m.ndim != 1:
            raise ValueError("l and m must be 1D tensors")
        if l.shape != m.shape:
            raise ValueError("l and m must have the same shape")
        if torch.any(m < 0):
            raise ValueError("This V1 evaluator only supports m >= 0")
        if torch.any(m > l):
            raise ValueError("Invalid harmonic indices: require m <= l")

        device = theta.device
        rdtype = theta.dtype
        cdtype = self.cdtype

        P = theta.numel()
        M = l.numel()

        if M == 0:
            return torch.empty((P, 0), dtype=cdtype, device=device)

        # x = cos(theta), shape [P]
        x = torch.cos(theta)

        # Clamp for numerical safety close to the poles.
        x = torch.clamp(x, -1.0, 1.0)

        # sin(theta) = sqrt(1 - x^2), clamped for float stability.
        sin_theta = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

        # Unique m values, sorted.
        unique_m, inverse_m = torch.unique(m, sorted=True, return_inverse=True)
        U = unique_m.numel()

        # Precompute exp(i m phi) for all unique m at once: [P, U]
        # Complex dtype is inferred from phi.
        exp_imphi = torch.exp(1j * phi[:, None] * unique_m.to(rdtype)[None, :])

        # Output
        Y = torch.empty((P, M), dtype=cdtype, device=device)

        # Constant log(4*pi)
        log_4pi = torch.as_tensor(
            math.log(4.0 * math.pi),
            dtype=rdtype,
            device=device,
        )

        # Loop over unique m only. Inside each m, all operations are vectorized over pixels.
        for u in range(U):
            mm = int(unique_m[u].item())

            mode_mask = (inverse_m == u)
            l_sel = l[mode_mask]  # [K]
            K = l_sel.numel()
            lmax_m = int(l_sel.max().item())

            # ------------------------------------------------------------
            # Compute associated Legendre P_l^m(x) for all needed l >= m
            # Shape target: [P, lmax_m - m + 1]
            # Index j corresponds to l = m + j
            # ------------------------------------------------------------

            ncols = lmax_m - mm + 1
            P_lm = torch.empty((P, ncols), dtype=rdtype, device=device)

            # P_m^m(x) = (-1)^m (2m-1)!! (1 - x^2)^{m/2}
            if mm == 0:
                p_mm = torch.ones_like(x)
            else:
                # Compute (2m-1)!! in log-space:
                # (2m-1)!! = (2m)! / (2^m m!)
                # log((2m-1)!!) = lgamma(2m+1) - m log(2) - lgamma(m+1)
                mm_t = torch.as_tensor(float(mm), dtype=rdtype, device=device)
                log_df = (
                    torch.lgamma(torch.as_tensor(2.0 * mm + 1.0, dtype=rdtype, device=device))
                    - mm_t * math.log(2.0)
                    - torch.lgamma(torch.as_tensor(mm + 1.0, dtype=rdtype, device=device))
                )
                coeff = torch.exp(log_df)

                # Condon-Shortley phase (-1)^m
                if mm % 2 == 1:
                    coeff = -coeff

                p_mm = coeff * (sin_theta ** mm)

            P_lm[:, 0] = p_mm

            if ncols >= 2:
                # P_{m+1}^m(x) = x (2m+1) P_m^m(x)
                p_m1m = x * (2 * mm + 1) * p_mm
                P_lm[:, 1] = p_m1m

                # Upward recurrence for l >= m+2
                # P_l^m(x) = ((2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m) / (l-m)
                p_lm2 = p_mm
                p_lm1 = p_m1m

                for ell in range(mm + 2, lmax_m + 1):
                    num = (2 * ell - 1) * x * p_lm1 - (ell + mm - 1) * p_lm2
                    den = (ell - mm)
                    p_cur = num / den
                    P_lm[:, ell - mm] = p_cur
                    p_lm2 = p_lm1
                    p_lm1 = p_cur

            # ------------------------------------------------------------
            # Normalization N_lm for selected l only
            # Y_lm = N_lm P_l^m(x) exp(i m phi)
            # with:
            # N_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
            # ------------------------------------------------------------
            l_sel_r = l_sel.to(rdtype)

            log_norm = 0.5 * (
                torch.log(2.0 * l_sel_r + 1.0)
                - log_4pi
                + torch.lgamma(l_sel_r - mm + 1.0)
                - torch.lgamma(l_sel_r + mm + 1.0)
            )
            norm = torch.exp(log_norm)  # [K]

            # Gather the needed l-columns from the dense-in-l block.
            cols = (l_sel - mm).to(torch.long)  # [K]
            P_sel = P_lm[:, cols]               # [P, K]

            # exp(i m phi) for this m
            phase = exp_imphi[:, u].unsqueeze(1)  # [P, 1]

            Y[:, mode_mask] = (P_sel * norm.unsqueeze(0)).to(cdtype) * phase

        return Y

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------

    def _compute_angles_from_nested_cell_ids(
        self,
        cell_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert nested HEALPix cell ids to angular coordinates.

        Parameters
        ----------
        cell_ids : torch.Tensor
            Nested cell ids, shape [N].

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            theta and phi in radians, both with shape [N].

        Notes
        -----
        This V1 backend uses healpy.pix2ang with nest=True.
        Later this should be replaced by healpix_geo for consistency.
        """
        pix_np = cell_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        theta_np, phi_np = hp.pix2ang(self.nside, pix_np, nest=True)
        theta = torch.as_tensor(theta_np, dtype=self.dtype, device=self.device)
        phi = torch.as_tensor(phi_np, dtype=self.dtype, device=self.device)
        return theta, phi

    def _compute_patch_center_vector(self) -> torch.Tensor:
        """
        Compute the mean unit direction of the patch.

        Returns
        -------
        torch.Tensor
            Unit vector of shape [3].
        """
        x = self.sin_theta * torch.cos(self.phi)
        y = self.sin_theta * torch.sin(self.phi)
        z = self.cos_theta
        vec = torch.stack([x.mean(), y.mean(), z.mean()], dim=0)
        norm = torch.linalg.norm(vec)
        if norm <= 0:
            raise RuntimeError("Degenerate patch center vector")
        return vec / norm

    def _vector_to_angles(self, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a 3D unit vector to (theta, phi).

        Parameters
        ----------
        vec : torch.Tensor
            Unit vector with shape [3].

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Colatitude and longitude in radians.
        """
        x, y, z = vec[0], vec[1], vec[2]
        theta = torch.arccos(torch.clamp(z, -1.0, 1.0))
        phi = torch.atan2(y, x)
        phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
        return theta, phi

    def _compute_patch_radius(self) -> torch.Tensor:
        """
        Compute the maximum angular distance from the patch center.

        Returns
        -------
        torch.Tensor
            Patch radius in radians.
        """
        x = self.sin_theta * torch.cos(self.phi)
        y = self.sin_theta * torch.sin(self.phi)
        z = self.cos_theta
        pts = torch.stack([x, y, z], dim=1)  # [N, 3]
        dot = torch.clamp(pts @ self._patch_center_vec, -1.0, 1.0)
        ang = torch.arccos(dot)
        return ang.max()

    # ---------------------------------------------------------------------
    # Spectral mode selection
    # ---------------------------------------------------------------------

    def _build_mode_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the local spectral index set according to mode='window'.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Degree and order arrays of shape [M].

        Notes
        -----
        V1 stores only m >= 0. This reduces memory compared to storing all
        negative and positive orders explicitly.
        """
        if self.mode != "window":
            raise ValueError("Only mode='window' is supported in V1")

        l_list = []
        m_list = []

        sin_r = float(torch.sin(self.patch_radius).item())

        for ell in range(self.lmax_compute + 1):
            m_cap = min(
                ell,
                max(
                    self.window_min_m,
                    int(math.ceil(self.window_alpha * ell * sin_r)) + self.window_margin_m,
                ),
            )
            for mm in range(m_cap + 1):  # V1 optimization: store only m >= 0
                l_list.append(ell)
                m_list.append(mm)

        if self.max_modes is not None and len(l_list) > self.max_modes:
            l_list = l_list[: self.max_modes]
            m_list = m_list[: self.max_modes]

        l = torch.tensor(l_list, dtype=torch.long, device=self.device)
        m = torch.tensor(m_list, dtype=torch.long, device=self.device)
        return l, m

    # ---------------------------------------------------------------------
    # Numerical helpers
    # ---------------------------------------------------------------------

    def _prepare_map_input(self, data: ArrayLike) -> Tuple[torch.Tensor, bool]:
        """
        Convert map input to a real tensor of shape [B, N].

        Returns
        -------
        (torch.Tensor, bool)
            Batched tensor and whether the original input was unbatched.
        """
        x = self._as_real_tensor(data, device=self.device, dtype=self.dtype)

        if x.ndim == 1:
            if x.shape[0] != self.n_pixels:
                raise ValueError("Input shape [N] does not match the transform geometry")
            return x.unsqueeze(0), True

        if x.ndim == 2:
            if x.shape[1] != self.n_pixels:
                raise ValueError("Input shape [B, N] does not match the transform geometry")
            return x, False

        raise ValueError("Input data must have shape [N] or [B, N]")

    def _prepare_coeff_input(
        self,
        coeffs: Union[AlmCoeffs, torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        """
        Convert coefficient input to a complex tensor of shape [B, M].

        Returns
        -------
        (torch.Tensor, bool)
            Batched complex tensor and whether the original input was unbatched.
        """
        if isinstance(coeffs, AlmCoeffs):
            alm = coeffs.alm
            if coeffs.l.shape != self.l.shape or coeffs.m.shape != self.m.shape:
                raise ValueError("Coefficient index set is incompatible with this transform")
        else:
            alm = coeffs

        if not torch.is_tensor(alm):
            alm = torch.as_tensor(alm)

        alm = alm.to(device=self.device, dtype=self.cdtype)

        if alm.ndim == 1:
            if alm.shape[0] != self.n_modes:
                raise ValueError("Coefficient shape [M] is incompatible with this transform")
            return alm.unsqueeze(0), True

        if alm.ndim == 2:
            if alm.shape[1] != self.n_modes:
                raise ValueError("Coefficient shape [B, M] is incompatible with this transform")
            return alm, False

        raise ValueError("Coefficients must have shape [M] or [B, M]")

    def _resolve_sigma(
        self,
        sigma: Optional[float],
        fwhm: Optional[float],
    ) -> float:
        """
        Resolve Gaussian smoothing sigma from either sigma or FWHM.

        Returns
        -------
        float
            Sigma in radians.
        """
        if sigma is None and fwhm is None:
            raise ValueError("Either sigma or fwhm must be provided")
        if sigma is not None and fwhm is not None:
            raise ValueError("Specify only one of sigma or fwhm")
        if sigma is not None:
            return float(sigma)
        return float(fwhm) / math.sqrt(8.0 * math.log(2.0))

    def _gaussian_beam(self, l: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Compute a Gaussian radial beam b_l = exp(-0.5 * l * (l + 1) * sigma^2).

        Parameters
        ----------
        l : torch.Tensor
            Degree indices.
        sigma : float
            Gaussian sigma in radians.

        Returns
        -------
        torch.Tensor
            Real beam values with shape matching `l`.
        """
        l_real = l.to(dtype=self.dtype, device=self.device)
        return torch.exp(-0.5 * l_real * (l_real + 1.0) * (sigma ** 2))

    # ---------------------------------------------------------------------
    # Static helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _infer_complex_dtype(dtype: torch.dtype) -> torch.dtype:
        """Infer the complex counterpart of a real torch dtype."""
        if dtype == torch.float32:
            return torch.complex64
        if dtype == torch.float64:
            return torch.complex128
        raise ValueError("Unsupported real dtype")

    @staticmethod
    def _resolve_device(
        device: Optional[Union[str, torch.device]],
    ) -> torch.device:
        """Resolve the execution device."""
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _as_long_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Convert input to a torch.long tensor."""
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    @staticmethod
    def _as_real_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert input to a real tensor."""
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)