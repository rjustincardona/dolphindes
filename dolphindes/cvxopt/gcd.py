"""
Methods for performing generalized constraint descent (GCD) for 
refining dual bounds of SharedProjQCQPs
"""

import numpy as np
import scipy.linalg as la
from .qcqp import SparseSharedProjQCQP


def merge_lead_constraints(QCQP: SparseSharedProjQCQP, merged_num: int = 2):
    """
    merge the first m constraints of QCQP into a single constraint
    also adjust the Lagrange multipliers so the dual value is the same
    
    Parameters
    ----------
    QCQP : SparseSharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; should be at least 2.
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    if merged_num < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if cstrt_num < merged_num:
        raise ValueError("Number of constraints insufficient for size of merge.")
    
    new_Pdiags = np.zeros((x_size,cstrt_num-merged_num+1), dtype=complex)
    new_lags = np.zeros(cstrt_num-merged_num+1, dtype=float)
    new_Pdiags[:,0] = QCQP.Pdiags[:,:merged_num] @ QCQP.current_lags[:merged_num]
    
    # normalize merged Pdiag
    Pnorm = la.norm(new_Pdiags[:,0])
    new_Pdiags[:,0] /= Pnorm
    new_lags[0] = Pnorm
    
    # put other constraints in
    new_Pdiags[:,1:] = QCQP.Pdiags[:,merged_num:]
    new_lags[1:] = QCQP.current_lags[merged_num:]
    
    # update QCQP
    if hasattr(QCQP, 'precomputed_As'):
        # updated precomputed_As
        QCQP.precomputed_As[merged_num-1] *= QCQP.current_lags[merged_num-1]
        for i in range(merged_num-1):
            QCQP.precomputed_As[merged_num-1] += QCQP.precomputed_As[i] * QCQP.current_lags[i]
        QCQP.precomputed_As[merged_num-1] /= Pnorm
        del QCQP.precomputed_As[:merged_num-1]
    
    if hasattr(QCQP, 'Fs'):
        new_Fs = np.zeros((x_size,cstrt_num-merged_num+1), dtype=complex)
        new_Fs[:,0] = QCQP.A2.conj().T @ (new_Pdiags[:,0].conj() * QCQP.s1)
        new_Fs[:,1:] = QCQP.Fs[:,merged_num:]
        QCQP.Fs = new_Fs
    
    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = None # in principle can merge dual derivatives but leave it undone for now
