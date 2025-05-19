import pytest
import numpy as np
from dolphindes.photonics.photonics import Photonics_TM_FDFD

def test_photonics_problem_initialization():
    # ...existing test setup...
    wvlgth = 1.0
    omega = 2 * np.pi / wvlgth
    
    prob = Photonics_TM_FDFD(omega)
    assert prob is not None  # Simple check to ensure constructor runs
    
    prob.chi = 3+1e-2j
    gpr = 40
    prob.dl = 1.0 / gpr
    Mx = My = 1*gpr

    
    Npmlsepx = Npmlsepy = Npmlx = Npmly = int(0.5*gpr)
    Nx = Mx + 2*(Npmlsepx + Npmlx)
    Ny = My + 2*(Npmlsepy + Npmly)
    des_mask = np.zeros((Nx,Ny), dtype=bool)
    des_mask[Npmlx+Npmlsepx:-(Npmlx+Npmlsepx) , Npmly+Npmlsepy:-(Npmly+Npmlsepy)] = True
    
    prob.setup_FDFD(Nx=Nx, Ny=Ny, Npmlx=Npmlx, Npmly=Npmly, des_mask=des_mask)
    
    ji = np.zeros((Nx,Ny), dtype=complex)
    ji[:,Npmly] = 1.0 / prob.dl # planewave line source
    
    ### ADD QCQP SETUP TEST ###