"""
Utilities for defining optimization geometries (cavities, etc.)
"""
import numpy as np

def embed_vector(des_mask, vector_r):
    """
    Function to embed a distribution vector(r), a 1D array defined only on the design region,
    into the full simulation domain, given by des_mask, a boolean array of the same shape as the simulation domain.

    Parameters
    ----------
    des_mask : np.ndarray
        Boolean mask defining the design region.
    vector_r : np.ndarray
        1D array of vector values defined only on the design region.
    """
    Nx, Ny = des_mask.shape
    vector_full = np.zeros((Nx, Ny), dtype=vector_r.dtype)
    vector_full[des_mask] = vector_r
    return vector_full