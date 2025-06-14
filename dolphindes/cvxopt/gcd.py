"""
Methods for performing generalized constraint descent (GCD) for 
refining dual bounds of SharedProjQCQPs

TODO: consider moving merge_lead_constraints and add_constraints into QCQP class methods
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from .qcqp import SparseSharedProjQCQP
from dolphindes.util import Sym


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


"""
TODO:
Use the QR decomposition to orthonormalize Pdiags
"""
def CRdot(v1: np.ndarray, v2: np.ndarray):
    """
    Computes the inner product of two complex vectors over a real field.
    In other words, the vectors have complex numbers but linear combination coefficients have to be real.
    This is the vector space for the complex QCQP constraints since Lagrangian multipliers are real.
    
    Parameters
    ----------
    v1 : np.ndarray
        vector1
    v2 : np.ndarray
        vector2

    Returns
    -------
    The inner product
    """
    return np.real(np.vdot(v1,v2))

def add_constraints(QCQP: SparseSharedProjQCQP, added_Pdiag_list: list, orthonormalize: bool=True):
    """
    method that adds new constraints into an existing QCQP. 
    
    Parameters
    ----------
    QCQP : SparseSharedProjQCQP
        QCQP for which the new constraints are added in.
    added_Pdiag_list : list
        List of 1d numpy arrays that are the new constraints to be added in
    orthonormalize : bool
        If true, assume that QCQP has orthonormal constraints and keeps it that way.
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    added_Pdiag_num = len(added_Pdiag_list)
    
    if not (QCQP.current_lags is None):
        new_lags = np.zeros(cstrt_num + added_Pdiag_num, dtype=float)
        new_lags[:cstrt_num] = QCQP.current_lags
    else:
        new_lags = None
    
    new_Pdiags = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
    new_Pdiags[:,:cstrt_num] = QCQP.Pdiags
    if not orthonormalize:
        for m,added_Pdiag in enumerate(added_Pdiag_list):
            new_Pdiags[:,cstrt_num+m] = added_Pdiag
    
    else:
        for m,added_Pdiag in enumerate(added_Pdiag_list):
            # do Gram-Schmidt orthogonalization for each added Pdiag
            for j in range(cstrt_num+m):
                added_Pdiag -= CRdot(new_Pdiags[:,j],added_Pdiag) * new_Pdiags[:,j]
            added_Pdiag /= la.norm(added_Pdiag)
            
            new_Pdiags[:,cstrt_num+m] = added_Pdiag
    
    # update QCQP
    if hasattr(QCQP, 'precomputed_As'):
        # updated precomputed_As
        for added_Pdiag in added_Pdiag_list:
            QCQP.precomputed_As.append(Sym(QCQP.A1 @ sp.diags_array(added_Pdiag, format='csr') @ QCQP.A2))
    
    if hasattr(QCQP, 'Fs'):
        new_Fs = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
        new_Fs[:,:cstrt_num] = QCQP.Fs
        new_Fs[:,cstrt_num:] = QCQP.A2.conj().T @ (new_Pdiags[:,cstrt_num:].conj().T * QCQP.s1).T
        QCQP.Fs = new_Fs
    
    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = None
