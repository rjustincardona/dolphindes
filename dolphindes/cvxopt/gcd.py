"""
Methods for performing generalized constraint descent (GCD) for 
refining dual bounds of SharedProjQCQPs
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from .qcqp import SparseSharedProjQCQP
from dolphindes.util import Sym


def merge_lead_constraints(QCQP: SparseSharedProjQCQP, m: int = 2):
    """
    merge the first m constraints of QCQP into a single constraint
    also adjust the Lagrange multipliers so the dual value is the same
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    if m < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if cstrt_num < m:
        raise ValueError("Number of constraints insufficient for size of merge.")
    
    new_Pdiags = np.zeros((x_size,cstrt_num-m+1), dtype=complex)
    new_lags = np.zeros(cstrt_num-m+1, dtype=float)
    new_Pdiags[:,0] = QCQP.Pdiags[:,:m] @ QCQP.current_lags[:m]
    
    # normalize merged Pdiag
    Pnorm = la.norm(new_Pdiags[:,0])
    new_Pdiags[:,0] /= Pnorm
    new_lags[0] = Pnorm
    
    # put other constraints in
    new_Pdiags[:,1:] = QCQP.Pdiags[:,m:]
    new_lags[1:] = QCQP.current_lags[m:]
    
    # update QCQP
    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = None
    # in principle can merge dual derivatives but leave it undone for now

    if hasattr(QCQP, 'precomputed_As'):
        # updated precomputed_As
        del QCQP.precomputed_As[:m-1]
        QCQP.precomputed_As[0] = Sym(QCQP.A1 @ sp.diags_array(QCQP.Pdiags[:,0], format='csr') @ QCQP.A2)

