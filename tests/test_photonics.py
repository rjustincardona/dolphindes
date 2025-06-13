import pytest
import numpy as np
from dolphindes.photonics.photonics import Photonics_TM_FDFD

def test_photonics_problem_initialization():
    # ...existing test setup...
    wvlgth = 1.0
    omega = 2 * np.pi / wvlgth

    chi = 3+1e-2j
    gpr = 40
    dl = 1.0 / gpr
    Mx = My = 1*gpr

    Npmlsepx = Npmlsepy = Npmlx = Npmly = int(0.5*gpr)
    Nx = Mx + 2*(Npmlsepx + Npmlx)
    Ny = My + 2*(Npmlsepy + Npmly)
    des_mask = np.zeros((Nx,Ny), dtype=bool)
    des_mask[Npmlx+Npmlsepx:-(Npmlx+Npmlsepx) , Npmly+Npmlsepy:-(Npmly+Npmlsepy)] = True

    ji = np.zeros((Nx,Ny), dtype=complex)
    ji[:,Npmly] = 1.0 / dl # planewave line source
    
    prob = Photonics_TM_FDFD(omega = omega, chi = chi, grid_size = (Nx, Ny), pml_size = (Npmlx, Npmly), dl = dl,
    des_mask = des_mask, ji=ji, chi_background=None, sparseQCQP=True, A0=None, s0=None, c0=0.0)
    
    assert prob is not None  # Simple check to ensure constructor runs
    
    ### ADD QCQP SETUP TEST ###