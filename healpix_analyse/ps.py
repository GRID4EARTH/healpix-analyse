import numpy as np

def ps(data, data_cross=None, plot_2D_fft=False):

    if data_cross is None:
        P2D = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
    else:
        P2D = np.abs(np.fft.fftshift(np.fft.fft2(data) * np.conj(np.fft.fft2(data_cross))))

    if plot_2D_fft:
        import matplotlib.pyplot as plt
        plt.imshow(P2D, norm='log')
        plt.colorbar(label='Power (log scale)', orientation='horizontal')
        plt.show()

    # Spatial frequency grids (cycles per unit length; NOT radians)
    ny, nx = P2D.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx))  # cycles per unit length (e.g., m^-1)
    fy = np.fft.fftshift(np.fft.fftfreq(ny))  # cycles per unit length (e.g., m^-1)
    fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
    fr = np.sqrt(fx2d**2 + fy2d**2)  # radial spatial frequency (cycles per unit length)

    # Radial binning
    nbins = min(nx, ny) // 2
    f_bins = np.linspace(0.0, fr.max(), nbins + 1)

    # Vectorized bin average of P2D over annuli
    fr_flat = fr.ravel()
    P_flat = P2D.ravel()
    bin_idx = np.digitize(fr_flat, f_bins) - 1  # -> [0, nbins-1]
    valid = (bin_idx >= 0) & (bin_idx < nbins)

    # Sum and count per bin, then mean
    sum_per_bin = np.bincount(bin_idx[valid], weights=P_flat[valid], minlength=nbins)
    cnt_per_bin = np.bincount(bin_idx[valid], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = sum_per_bin / cnt_per_bin
    Pk[cnt_per_bin == 0] = np.nan  # empty bins

    # Bin centers
    f_centers = 0.5 * (f_bins[1:] + f_bins[:-1])

    return f_centers, Pk / (nx*ny)