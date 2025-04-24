from dolphindes.maxwell import TM_FDFD
import numpy as np 
import scipy.sparse as sp 

def test_maxwell_TM():
    wvlgth = 1.0 
    omega = 2 * np.pi / wvlgth
    gpr = 50 
    dl = 1 / gpr 
    Nx, Ny = int(3.0*gpr), int(3.0*gpr) 
    Npmlx, Npmly = int(0.5*gpr), int(0.5*gpr)

    print('Testing agreement between TM_FDFD and G_ba for a dipole source')
    simulation = TM_FDFD(omega, Nx, Ny, Npmlx, Npmly, dl)
    Ez_simulation = simulation.get_TM_dipole_field(Nx//2, Ny//2)

    a_mask = np.zeros((Nx, Ny), dtype=bool)
    a_mask[Nx//4:-Nx//4, Ny//4:-Ny//4] = True 
    G_ba = simulation.get_TM_G_ba(a_mask, a_mask)
    sourcegrid = np.zeros((Nx, Ny), dtype=complex)
    sourcegrid[Nx//2, Ny//2] = 1.0/dl/dl
    Ez = ((-1j * simulation.k / simulation.ETA_0)**-1) * G_ba @ sourcegrid[a_mask]
    Ez_large = np.zeros((Nx, Ny), dtype=complex)
    Ez_large[a_mask] = Ez

    Ez_sim_large = np.zeros((Nx, Ny), dtype=complex)
    Ez_sim_large[a_mask] = Ez_simulation[a_mask]

    assert np.allclose(Ez_large, Ez_sim_large, atol=1e-6), "Ez from G_ba does not match Ez from simulation"

    print('Testing Ginv and G_ba agreement')
    Ginv, M = simulation.get_Gaainv(a_mask)
    assert type(Ginv) is sp.csc_array 
    assert type(M) is sp.csc_array
    assert Ginv.shape == (np.sum(a_mask), np.sum(a_mask)), "Ginv shape mismatch"

    assert np.allclose(Ginv @ G_ba, np.eye(np.sum(a_mask)), atol=1e-6), "Ginv @ G_ba does not equal identity matrix"
