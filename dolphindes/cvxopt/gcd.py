"""
Methods for performing general constraint descent (GCD) for 
refining dual bounds of SharedProjQCQPs

TODO: consider moving merge_lead_constraints and add_constraints into QCQP class methods
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from .qcqp import SparseSharedProjQCQP
from dolphindes.util import Sym, CRdot


def run_gcd(QCQP: SparseSharedProjQCQP, 
            max_cstrt_num: int = 10, orthonormalize: bool=True,
            max_gcd_iter_num=50, gcd_iter_period=5, gcd_tol=1e-2):
    """
    Perform generalized constraint descent to gradually refine dual bound on QCQP.
    TODO: formalize optimization and convergence parameters.
    Parameters
    ----------
    QCQP : SparseSharedProjQCQP
        The SharedProjQCQP for which we compute and refine dual bounds.
    max_cstrt_num : int, optional
        The maximum constraint number for QCQP. The default is 10.
    orthonormalize : bool, optional
        Whether or not to orthonormalize the constraint projectors. The default is True.
    """
    # get to feasible point
    QCQP.current_lags = QCQP.find_feasible_lags()
    if orthonormalize:
        # orthonormalize QCQP
        # informally checked for correctness
        x_size, cstrt_num = QCQP.Pdiags.shape
        realext_Pdiags = np.zeros((2*x_size,cstrt_num), dtype=float)
        realext_Pdiags[:x_size,:] = np.real(QCQP.Pdiags)
        realext_Pdiags[x_size:,:] = np.imag(QCQP.Pdiags)
        realext_Pdiags_Q, realext_Pdiags_R = la.qr(realext_Pdiags, mode='economic')
        QCQP.Pdiags = realext_Pdiags_Q[:x_size,:] + 1j*realext_Pdiags_Q[x_size:,:]
        QCQP.current_lags = realext_Pdiags_R @ QCQP.current_lags
        QCQP.compute_precomputed_values()
    
    ## gcd loop
    gcd_iter_num = 0
    gcd_prev_dual = np.inf
    while True:
        gcd_iter_num += 1
        # solve current dual problem
        # TODO: add QCQP convergence params
        QCQP.solve_current_dual_problem('newton', init_lags=QCQP.current_lags)
        print(f'At GCD iteration #{gcd_iter_num}, best dual bound found is {QCQP.current_dual}.')
        
        ## termination conditions
        if gcd_iter_num > max_gcd_iter_num:
            break
        if gcd_iter_num % gcd_iter_period==0:
            if gcd_prev_dual - QCQP.current_dual < gcd_tol * abs(gcd_prev_dual):
                break
            gcd_prev_dual = QCQP.current_dual
        
        ## generate max dualgrad constraint
        maxViol_Pdiag = (2*QCQP.s1 - (QCQP.A1.conj().T @ QCQP.current_xstar)) * (QCQP.A2 @ QCQP.current_xstar).conj()
        
        ## generate min A eig constraint
        minAeigv, minAeigw = QCQP._get_PSD_penalty(QCQP.current_lags)
        minAeig_Pdiag = (QCQP.A1.conj().T @ minAeigv) * (QCQP.A2 @ minAeigv).conj()
        # use the same relative weights for minAeig_Pdiag as maxViol_Pdiag
        minAeig_Pdiag /= np.sqrt(np.real(minAeig_Pdiag.conj() * minAeig_Pdiag))
        minAeig_Pdiag * np.sqrt(np.real(maxViol_Pdiag.conj() * maxViol_Pdiag))
        # informally checked that minAeigw increases when increasing multiplier of minAeig_Pdiag
        
        ## add new constraints
        QCQP.add_constraints([maxViol_Pdiag, minAeig_Pdiag])
        # informally checked that new constraints are added in orthonormal fashion
    
        ## merge old constraints if necessary
        if cstrt_num > max_cstrt_num:
            QCQP.merge_lead_constraints(merged_num = cstrt_num-max_cstrt_num+1)
    
