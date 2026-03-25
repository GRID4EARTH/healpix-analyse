from scipy.interpolate import griddata
import numpy as np

def resample_to_latlon_grid(lat, lon, data, method='linear'):
    """Resample HEALPix data onto a regular latitude-longitude grid using interpolation."""
    lon_grid, lat_grid = np.meshgrid(np.linspace(lon.min(), lon.max(), data.shape[1]),
                                     np.linspace(lat.min(), lat.max(), data.shape[0]))
    points = np.column_stack((lat.flatten(), lon.flatten()))
    data_resampled = griddata(points, 
                              data.flatten(), 
                              (lat_grid.flatten(), lon_grid.flatten()), 
                              method=method,
                              fill_value=data.mean())
    data_resampled = data_resampled.reshape(lon_grid.shape)
    return data_resampled