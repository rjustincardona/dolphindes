"""
Dual Problem Interface

"""

__all__ = ['SparseSharedProjQCQP'] 

import numpy as np 
import scipy.sparse as sp
import sksparse.cholmod 

class SparseSharedProjQCQP():
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is 
    max_x x^dagger A0 x + 2 Re (x^dagger s0) + c
    s.t.  Re(x^dagger A1 P_j A_2 x + 2 Re (x^dagger A_2^dagger P_j^dagger s1) + c1) = 0

    for a list of projection matrices P_j. 

    Attributes
    ----------
    A0 : scipy.sparse.csc_array
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c : float
        The constant c in the QCQP.
    A1 : scipy.sparse.csc_array
        The matrix A1 in the QCQP.
    A2 : scipy.sparse.csc_array
        The matrix A2 in the QCQP.
    s1 : np.ndarray
        The vector s1 in the QCQP.
    c1 : float
        The constant c1 in the QCQP.
    projections_diags : np.ndarray
        The diagonal elements of the projection matrices P_j.  
    verbose : float
        The verbosity level for debugging and logging.
    Achofac : sksparse.cholmod.Cholesky
        The Cholesky factorization of the total A matrix, which is updated when needed.
    current_point : tuple
        The current dual solution, which is only updated when the dual problem is solved. Tuple of (dual value, gradient, hessian).
    """
    def __init__(self, A0: sp.csc_array, s0: np.ndarray, c: float, A1: sp.csc_array, A2: sp.csc_array, 
                 s1: np.ndarray, c1: float, projections_diags: np.ndarray, verbose: float = 0):
        self.A0 = A0
        self.s0 = s0
        self.s1 = s1
        self.c = c
        self.A1 = A1
        self.A2 = A2
        self.c1 = c1
        self.verbose = verbose
        self.projections_diags = projections_diags
        self.Achofac = None
        self.current_dual = None 

        self._update_chofac()

    def _Sym(self, A: sp.csc_array) -> sp.csc_array:
        """Gets the symmetric part of A"""
        return (A + A.T.conj()) / 2
    
    def _add_projectors(self, lags: np.ndarray) -> np.ndarray:
        """Combine the lagrange multipliers and the projectors into a joint projector

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.
        
        Returns
        -------
        np.ndarray
            The diagonal elements of the combined projector matrix sum_j lag[j] P_j 
        """
        return np.sum(lags[:, np.newaxis] * self.projections_diags, axis=0)

    def _get_total_A(self, P: sp.csc_array) -> sp.csc_array:
        """Gets the total A matrix for the QCQP = A0 + sum_j lag[j] A1 P_j A2 given sum_j lag[j] P_j"""
        return self.A0 + self._Sym(self.A1 @ P @ self.A2)
    
    def _get_total_S(self, P: sp.csc_array) -> np.ndarray:
        """Gets the total S vector for the QCQP = s0 + sum_j lag[j] P_j^dagger s1 given P = sum_j lag[j] P_j"""
        return self.s0 + self.A2.T.conj() @ P.T.conj() @ self.s1

    def _update_chofac(self):
        """Updates the cholesky factorization of the total A
        
        Returns
        -------
        sksparse.cholmod.Cholesky
            The updated cholesky factorization of the total A matrix.
        """
        random_lags = np.random.rand(len(self.projections_diags))
        P = self._add_projectors(random_lags)
        A = self._get_total_A(sp.csc_array(sp.diags(P, format='csc')))
        if self.verbose > 1: print(f"analyzing A of format and shape {type(A)}, {A.shape} and # of nonzero elements '{A.count_nonzero()}")
        self.Achofac = sksparse.cholmod.analyze(A)
        return self.Achofac

    def _get_xstar(self, A: sp.csc_array, S: np.ndarray, get_xgrad: bool = False):
        """For a total A and S, solve for xstar using the cholesky factorization of A
        
        For a given set of Lagrange multipliers, which define A and S, the x_star that maximizes the Lagrangian 
        (and therefore defines the dual) is given by A x_star = S. Since the symbolic Cholesky factorization 
        of A is known, because the sparsity pattern of A is always known, this solution can be found very
        efficiently. 

        Returns
        -------
        Acho : sksparse.cholmod.Cholesky
            The cholesky factorization of A.
        x_star : np.ndarray
            The solution x_star to the equation A x_star = S.
        grad_x : np.ndarray
            The gradient of the Lagrangian with respect to x_star, if requested (otherwise empty array)

        """
        Acho = self.Achofac.cholesky(A)
        x_star = Acho.solve_A(S)
        Pdiag_list = self.projections_diags
        grad_x = np.zeros(Pdiag_list.shape[0])
        
        if get_xgrad:
            # grad_x_mat is a matrix containing the vectors b that require solving Ax=b for each lagrange multiplier
            # this is useful for the Hessian calculation 
            grad_x_mat = -self.A1 @ (x_star[:, None] * Pdiag_list[None, :]) + (self.s1[:, None] * Pdiag_list[None, :])
            grad_x = Acho.solve_A(grad_x_mat)
        
        return Acho, x_star, grad_x
        
    def get_dual(self, lags: np.ndarray, get_grad: bool = False, get_hess: bool = False, penalty_s: list = None) -> tuple[float, np.ndarray, np.ndarray]:
        """Gets the dual value and the derivatives of the dual with respect to Lagrange multipliers.

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.
        get_grad : bool, optional
            Whether to return the gradient of the dual with respect to the lagrange multipliers. Default is False.
        get_hess : bool, optional
            Whether to return the Hessian of the dual with respect to the lagrange multipliers. Default is False.
        penalty_s : list, optional
            A list of penalty vectors for the PSD boundary. If provided, the dual value and gradients will include a penalty term. Default is None.
        
        Returns
        -------
        dualval : float
            The value of the dual function.
        grad : np.ndarray
            The gradient of the dual function with respect to the lagrange multipliers, if requested.
        hess : np.ndarray
            The Hessian of the dual function with respect to the lagrange multipliers, if requested.
        """
        if get_hess: raise NotImplementedError("SparseSharedProjQCQP cannot return the Hessian yet. Use a first order method.")
        
        grad, hess = [], []

        P_diag = self._add_projectors(lags)
        P = sp.csc_array(sp.diags(P_diag))
        A = self._get_total_A(P)
        S = self._get_total_S(P)

        Acho, xstar, grad_x = self._get_xstar(A, S, get_xgrad = get_hess) # grad_x is needed for calculating the hessian. xstar is sufficient for the gradient. 

        dualval = self.c + np.real(xstar.conjugate() @ A @ xstar)

        if get_grad: # This is grad_lambda (not grad_x)
            # slow way 
            grad = np.zeros(self.projections_diags.shape[0])
            for i in range(len(grad)): 
                # print(-np.real(xstar.conj() @ (self.projections_diags[i] * xstar)))
                # print(2*np.real(xstar.conj() @ (self.projections_diags[i] * self.s1)))
                grad[i] = -np.real(xstar.conj() @ (self.projections_diags[i] * xstar)) + 2*np.real(xstar.conj() @ (self.projections_diags[i] * self.s1))

            # fast way
            # Px = self.projections_diags * xstar[:, np.newaxis]  # Efficiently apply each projection to xstar
            # grad = -np.real(xstar.conj() @ self.A1 @ Px) + 2 * np.real(self.s1.conj() @ Px)

        # Boundary penalty for the PSD boundary 
        if penalty_s is not None:
            assert False
            penalty_matrix = np.column_stack(penalty_s)
            A_inv_penalty = Acho.solve_A(penalty_matrix)
            dualval += np.sum(np.real(A_inv_penalty.conj() * penalty_matrix)) # multiplies columns with columns, sums all at once
        
            if get_grad: 
                pass 
                #P_ps = self.projections_diags * penalty_matrix

        return dualval, grad, hess 

    def solve_current_dual_problem(self, method: str) -> None:
        """Solves the current dual problem using the specified method.

        Parameters
        ----------
        method : str
            The method to use for solving the dual problem. 'newton' or 'bfgs'
        """
        raise NotImplementedError("SparseSharedProjQCQP cannot solve the dual problem yet.")


class DenseSharedProjQCQP():
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is 
    max_x x^dagger A0 x + 2 Re (x^dagger s0) + c
    s.t.  Re(x^dagger A1 P_j x + 2 Re (x^dagger P_j^dagger s1) + c) = 0

    for a list of projection matrices P_j. 

    Attributes
    ----------
    A0 : np.ndarray 
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c : float
        The constant c in the QCQP.
    A1 : np.ndarray
        The matrix A1 in the QCQP.
    projections_diags : np.ndarray
        The diagonal elements of the projection matrices P_j. The first one should always be one such that A0 + lambda A1 P_0 is positive semidefinite for sufficiently large constant lambda. 
    """
    def __init__(self, A0, s0, c, A1, s1, projections_diags: np.ndarray, verbose: float = 0):
        self.A0 = A0
        self.s0 = s0
        self.s1 = s1
        self.c = c
        self.A1 = A1
        self.verbose = verbose
        self.projections_diags = projections_diags
        self.current_dual = None 

    def _add_projectors(self, lags: np.ndarray) -> np.ndarray:
        """Combine the lagrange multipliers and the projectors into a joint projector

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.
        
        Returns
        -------
        np.ndarray
            The diagonal elements of the combined projector matrix sum_j lag[j] P_j 
        """
        return np.sum(lags[:, np.newaxis] * self.projections_diags, axis=0)

    def _get_total_A(self, P: sp.csc_array) -> sp.csc_array:
        """Gets the total A matrix for the QCQP = A0 + sum_j lag[j] A1 P_j given sum_j lag[j] P_j"""
        return self.A0 + self.A1 @ P
    
    def _get_total_S(self, P: sp.csc_array) -> np.ndarray:
        """Gets the total S vector for the QCQP = s0 + sum_j lag[j] P_j^dagger s1 given P = sum_j lag[j] P_j"""
        return self.s0 + P.T.conj() @ self.s1

    def _get_xstar(self, A: sp.csc_array, S: np.ndarray, get_xgrad: bool = False):
        """For a total A and S, solve for xstar using the cholesky factorization of A
        
        For a given set of Lagrange multipliers, which define A and S, the x_star that maximizes the Lagrangian 
        (and therefore defines the dual) is given by A x_star = S. Since the symbolic Cholesky factorization 
        of A is known, because the sparsity pattern of A is always known, this solution can be found very
        efficiently. 

        Returns
        -------
        Acho : sksparse.cholmod.Cholesky
            The cholesky factorization of A.
        x_star : np.ndarray
            The solution x_star to the equation A x_star = S.
        grad_x : np.ndarray
            The gradient of the Lagrangian with respect to x_star, if requested (otherwise empty array)

        """
        pass 
        
    def get_dual(self, lags: np.ndarray, get_grad: bool = False, get_hess: bool = False, penalty_s: list = None) -> tuple[float, np.ndarray, np.ndarray]:
        """Gets the dual value and the derivatives of the dual with respect to Lagrange multipliers.

        Parameters
        ----------

        Returns
        -------
        """
        pass 

    def solve_current_dual_problem(self, method: str) -> None:
        pass 