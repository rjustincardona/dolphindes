import numpy as np
import scipy.sparse as sp
import time, os, sys

package_path = os.path.abspath('../dolphindes')
if package_path not in sys.path:
    sys.path.append(package_path)
from dolphindes import photonics, geometry as geo


def bound(n):
    wavelength = 1.0 # Dolphindes uses dimensionless units. 
    omega = 2 * np.pi / wavelength
    chi = 4+1e-4j # Design material 
    px_per_length = n # pixels per length unit. If wavelength = 1.0, then this is pixels per wavelength.
    dl = 1/px_per_length 
    Npmlsep = int(0.5 / dl) # gap between design region and PML. Not required to be defined, it is just convenient.
    Npmlx, Npmly = int(0.5 / dl), int(0.5 / dl) # PML size.
    Mx, My = int(0.5 / dl), int(0.5 / dl) # design mask size 
    Dx = int(0.1 / dl) # distance from the design region to the source region.
    Nx, Ny = int(Npmlx*2 + Npmlsep*2 + Dx + Mx), int(Npmly*2 + Npmlsep*2 + My) # grid size. This includes the pml layer!

    cx, cy = Npmlx + Npmlsep, Ny//2

    ji = np.zeros((Nx, Ny), dtype=complex) # current density
    ji[cx, cy] = 1.0/dl/dl # a delta function source in 2D is approximated by amplitude 1/dl/dl so that integration int(ji)dxdy = 1.0. 
    design_mask = np.zeros((Nx, Ny), dtype=bool) # design mask
    design_mask[Npmlx + Npmlsep + Dx: Npmlx + Npmlsep + Dx + Mx, Npmly + Npmlsep: Npmly + Npmlsep + My] = True # design mask
    ndof = np.sum(design_mask) # number of degrees of freedom in the design region

    chi_background = np.zeros((Nx, Ny), dtype=complex) # background material
    chi_background[:, :] = 0


    # Setup geometry
    geometry = geo.CartesianFDFDGeometry(
        Nx=Nx, Ny=Ny, Npmlx=Npmlx, Npmly=Npmly, dx=dl, dy=dl
    )

    ldos_problem = photonics.Photonics_TM_FDFD(
        omega=omega, geometry=geometry, chi=chi,
        des_mask=design_mask, ji=ji, chi_background=chi_background, 
        sparseQCQP=True
    )

    ei = ldos_problem.get_ei(ji, update=True) # update = true sets the ei to the source field. Not required if you just need to do a Maxwell solve. 
    vac_ldos = -np.sum(1/2 * np.real(ji.conj() * ei) * dl * dl)

    ei_design = ei[ldos_problem.des_mask] # restrict the field to the design region
    c0 = vac_ldos
    s0_p = - (1/4) * 1j * omega * ei_design.conj() #* dl * dl  # the dl*dl factor is not needed here, as set_objective will take care of it.
    A0_p = sp.csc_array(np.zeros((ndof, ndof), dtype=complex))

    ldos_problem.set_objective(s0=s0_p, A0=A0_p, c0=vac_ldos, denseToSparse=True)
    ldos_problem.setup_QCQP(Pdiags = 'global', verbose=1) # verbose has a few levels. 0 is silent, 1 is basic output, 2 is more verbose, 3 is very verbose.
    import copy
    gcd_QCQP = copy.deepcopy(ldos_problem.QCQP)


    from dolphindes.cvxopt import gcd

    ### now compare with tightening the bounds using GCD

    ## gcd parameters, play around and see how the result changes

    # maximum number of QCQP constraints before merging, larger values may lead to tighter final bounds but makes GCD slower
    max_cstrt_num = 10

    # maximum number of GCD iterations
    max_gcd_iter_num = 50

    # check to see how much the bound improved after gcd_iter_period number of GCD iterations
    gcd_iter_period = 5

    # relative tolerance for required minimum improvement of bounds or GCD terminates
    gcd_tol = 1e-2

    t = time.time()

    gcd_params = gcd.GCDHyperparameters(
        max_proj_cstrt_num=max_cstrt_num,
        orthonormalize=True,
        opt_params=None,
        max_gcd_iter_num=max_gcd_iter_num,
        gcd_iter_period=gcd_iter_period,
        gcd_tol=gcd_tol
    )
    gcd_QCQP.run_gcd(gcd_params=gcd_params)
    print(f'gcd took time {time.time()-t} to reach {gcd_QCQP.current_dual} of pixel dual.')

bound(20)
