#!/usr/bin/env python
# coding: utf-8

# # Computing Bounds on the Local Density of States using General Constraint Descent
# 
# This notebook demonstrates the use of general constraint descent (GCD) for evaluating dual bounds, using local density of states (LDOS) maximization as the example problem. For more background on LDOS optimization and bounds, see LDOS.ipynb. 
# 
# The tightest dual bound for a photonic inverse design problem is found by imposing all possible local power conservation constraints in the corresponding field optimization QCQP. Unfortunately, due to the large number of constraints, evaluating this tightest dual bound can be extremely expensive.
# 
# The basic idea of GCD is to approximate the tightest dual bound with a smaller, more manageable number of constraints. This is pursued by iteratively adding new constraints to the QCQP to tighten the bounds, and merging old constraints to keep the total number of constraints fixed. 
# 
# GCD is implemented for all shared projection QCQPs in dolphindes, and this notebook demonstrates the high-level use of the available GCD functionality. For more mathematical details on GCD, see Appendix B of https://arxiv.org/abs/2504.14083.

# In[1]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys, time, os

package_path = os.path.abspath('../dolphindes')
if package_path not in sys.path:
    sys.path.append(package_path)

from dolphindes import photonics


# In[2]:


# First, let's define the relevant parameters for the simulation. 

wavelength = 1.0 # Dolphindes uses dimensionless units. 
omega = 2 * np.pi / wavelength
chi = 4+1e-4j # Design material 
px_per_length = 40 # pixels per length unit. If wavelength = 1.0, then this is pixels per wavelength.
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

# plt.matshow(design_mask + np.real(ji)*dl*dl) # visualize where the mask and the source are


# In[3]:


# Let's now initiate the photonics TM FDFD class. Leave the objective empty for now, let's use the class to compute the source field first. 
# s0 and A0 do not have to be passed now, and in general don't need to be passed to do some EM calculations. 
ldos_problem = photonics.Photonics_TM_FDFD(omega = omega, chi = chi, grid_size = (Nx, Ny), pml_size = (Npmlx, Npmly), dl = dl,
    des_mask = design_mask, ji=ji, chi_background=chi_background, sparseQCQP=True)

# You can print the ldos problem to see the attributes.
print(ldos_problem)

ei = ldos_problem.get_ei(ji, update=True) # update = true sets the ei to the source field. Not required if you just need to do a Maxwell solve. 
# plt.imshow(np.real(ei), cmap='bwr')

vac_ldos = -np.sum(1/2 * np.real(ji.conj() * ei) * dl * dl)
print("Vacuum LDOS: ", vac_ldos)


# In[4]:


# Now let's set s0. We need to restrict ei to the design region. 
ei_design = ei[ldos_problem.des_mask] # restrict the field to the design region
c0 = vac_ldos
s0_p = - (1/4) * 1j * omega * ei_design.conj() * dl * dl
A0_p = sp.csc_array(np.zeros((ndof, ndof), dtype=complex))

# We set the objective with set_objective(). 
ldos_problem.set_objective(s0=s0_p, A0=A0_p, c0=vac_ldos, denseToSparse=True)


# In[5]:


# We are ready to set up the QCQP for calculating limits. We will use Pdiags = 'global': this represents two constraints (extinction and real power global conservation). We will show how to refine these constraints below, or you may pass Pdiags = 'local' to directly do the local problem (often slower).
ldos_problem.setup_QCQP(Pdiags = 'global', verbose=0) # verbose has a few levels. 0 is silent, 1 is basic output, 2 is more verbose, 3 is very verbose.

# # get a copy of the ldos_problem QCQP for testing and comparing with GCD
import copy
gcd_QCQP = copy.deepcopy(ldos_problem.QCQP)


# In[6]:


## iterative splitting
# method = 'bfgs'
# ldos_problem.QCQP.solve_current_dual_problem(method=method)
# results = []
# result_counter = 0
# t = time.time()
# for result in ldos_problem.QCQP.iterative_splitting_step(method=method): # When we reach pixel level constraints, the generator will return and stop this loop.
#     result_counter += 1
#     num_constr = ldos_problem.QCQP.get_number_constraints()
#     print(f'at step {result_counter}, number of constraints is {num_constr}, bound is {result[0]}')

#     results.append((num_constr, result[0]))
# print(f'Newton iterative splitting took {time.time()-t}s to reach pixel level.')


# # In[7]:


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

gcd_QCQP.run_gcd(max_cstrt_num=max_cstrt_num, max_gcd_iter_num=max_gcd_iter_num,
                 gcd_iter_period=gcd_iter_period, gcd_tol=gcd_tol)
# print(f'gcd took time {time.time()-t} to reach {gcd_QCQP.current_dual/ldos_problem.QCQP.current_dual} of pixel dual.')
print(f'gcd took time {time.time()-t} to reach {gcd_QCQP.current_dual} at pixel dual.')

# n = int(np.sqrt(gcd_QCQP.current_xstar.shape[0]))
# fields = np.reshape(gcd_QCQP.current_xstar, (n, n))
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# im_real = ax[0].imshow(np.abs(fields), cmap="viridis")
# im_imag = ax[1].imshow(np.angle(fields), cmap="twilight")
# cbar_real = fig.colorbar(im_real, ax=ax[0], fraction=0.046, pad=0.04)
# cbar_imag = fig.colorbar(im_imag, ax=ax[1], fraction=0.046, pad=0.04)
# for a in ax:
#     a.axis("off")
#     ax[0].set_title(r"Field magnitude")
#     ax[1].set_title(r"Field phase")
# plt.show()
