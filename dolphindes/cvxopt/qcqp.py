"""
Dual Problem Interface

"""

__all__ = ['SharedProjQCQP'] 

import numpy as np 
import scipy.sparse as sp

class SharedProjQCQP():
    def __init__(self, A0, s, c, A1, semidefinite_P, projections_diags: np.ndarray):
        self.A0 = A0
        self.s = s 
        self.c = c
        self.A1 = A1

        self.semidefinite_P = semidefinite_P
        self.projections_diags = projections_diags

        self.current_dual = None 

    def get_dual(self, lags: np.array, get_grad: bool = False, get_hess: bool = False) -> None:
        if get_hess:
            raise NotImplementedError("Hessian not implemented yet")
        if get_grad:
            raise NotImplementedError("Gradient not implemented yet")
        
        totalA1 = self.A0 + self.A1 @ (lags[0] * self.semidefinite_P + np.sum(lags[1:, np.newaxis] * self.projections_diags, axis=0))
        xstar = np.ones(totalA1.shape[0])
        dualval = self.c + xstar.conjugate() @ totalA1 @ xstar

        if not get_grad:
            return dualval 
        elif get_grad and not get_hess:
            return dualval, None
        elif get_grad and get_hess:
            return dualval, None, None
        else:
            raise NotImplementedError("This should not happen")

    def solve_current_dual_problem(self):
        # define mineig, etc 
        # Send it to an optimizer 
        pass 

    def solve_gcd(self):
        pass 

class SparseSharedProjQCQP(SharedProjQCQP):
    def __init__(self, A, s, c, Q, projections_diags: np.ndarray):
        super().__init__(A, s, c, Q, projections_diags)

    def solve_current_problem(self):
        pass