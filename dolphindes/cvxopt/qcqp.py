"""
Dual Problem Interface

"""

__all__ = ['SparseSharedProjQCQP'] 

import numpy as np 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod 
from .optimization import BFGS
from collections import namedtuple
from typing import Optional, Dict, Any, Tuple # For type hinting the new method


class SparseSharedProjQCQP():
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is 
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j. 

    Attributes
    ----------
    A0 : scipy.sparse.csc_array
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c0 : float
        The constant c in the QCQP.
    A1 : scipy.sparse.csc_array
        The matrix A1 in the QCQP.
    A2 : scipy.sparse.csc_array
        The matrix A2 in the QCQP.
    s1 : np.ndarray
        The vector s1 in the QCQP.
    Pdiags : np.ndarray
        The diagonal elements of the projection matrices P_j, as columns of a matrix Pdiags
        The second column j should always be one such that A0 + lambda A1 P_j is positive semidefinite for sufficiently large constant lambda.
    verbose : float
        The verbosity level for debugging and logging.
    Achofac : sksparse.cholmod.Cholesky
        The Cholesky factorization of the total A matrix, which is updated when needed.
    current_dual : float
        The current dual solution, which is only updated when the dual problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only updated when the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only updated when the dual problem is solved. 
    """
    def __init__(self, A0: sp.csc_array, s0: np.ndarray, c0: float, A1: sp.csc_array, A2: sp.csc_array, 
                 s1: np.ndarray, Pdiags: np.ndarray, verbose: float = 0):
        self.A0 = A0
        self.s0 = s0
        self.s1 = s1
        self.c0 = c0
        self.A1 = A1
        self.A2 = A2
        self.verbose = verbose
        self.Pdiags = Pdiags
        self.Achofac = None
        self.current_dual = None 
        self.current_grad = None 
        self.current_hess = None

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

        # lags is (N,), Pdiags is (N, M)
        # lags[:, np.newaxis] makes lags (N, 1)
        # Broadcasting then multiplies each row of Pdiags by the corresponding lag
        # return np.sum(lags[:, np.newaxis] * self.Pdiags, axis=0)
        return self.Pdiags @ lags

    def _get_total_A(self, P: sp.csc_array) -> sp.csc_array:
        """Gets the total A matrix for the QCQP = A0 + sum_j lag[j] A1 P_j A2 given sum_j lag[j] P_j"""
        return self.A0 + self._Sym(self.A1 @ P @ self.A2)
    
    def _get_total_S(self, P: sp.csc_array) -> np.ndarray:
        """Gets the total S vector for the QCQP = s0 + sum_j lag[j] P_j^dagger s1 given P = sum_j lag[j] P_j"""
        return self.s0 + self.A2.T.conj() @ P.T.conj() @ self.s1 

    def _update_chofac(self) -> sksparse.cholmod.Factor:
        """Updates the cholesky factorization of the total A
        
        Returns
        -------
        sksparse.cholmod.Cholesky
            The updated cholesky factorization of the total A matrix.
        """
        random_lags = np.random.rand(self.Pdiags.shape[1])
        P = self._add_projectors(random_lags)
        A = self._get_total_A(sp.diags_array(P, format='csc'))
        if self.verbose > 1: print(f"analyzing A of format and shape {type(A)}, {A.shape} and # of nonzero elements '{A.count_nonzero()}")
        self.Achofac = sksparse.cholmod.analyze(A)
        return self.Achofac

    def _get_xstar(self, A: sp.csc_array, S: np.ndarray, get_xgrad: bool = False) -> tuple[sksparse.cholmod.Factor, np.ndarray, np.ndarray]:
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
        grad_x = [] #np.zeros(Pdiag_list.shape[0])

        if get_xgrad:
            # grad_x_mat is a matrix containing the vectors b that require solving Ax=b for each lagrange multiplier
            # this is useful for the Hessian calculation. 
            # Warning("get_xgrad (needed for Hessian) is not tested!")
            # grad_x_mat = -self.A1 @ (x_star[:, None] * Pdiag_list[None, :]) + (self.s1[:, None] * Pdiag_list[None, :])
            # grad_x = Acho.solve_A(grad_x_mat)
            raise NotImplementedError("get_xgrad is not implemented yet. Use a first order method.")
    
        return Acho, x_star, grad_x
        
    def get_dual(self, lags: np.ndarray, get_grad: bool = False, get_hess: bool = False, penalty_vectors: list = []) -> tuple[float, np.ndarray, np.ndarray]:
        """Gets the dual value and the derivatives of the dual with respect to Lagrange multipliers.

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.
        get_grad : bool, optional
            Whether to return the gradient of the dual with respect to the lagrange multipliers. Default is False.
        get_hess : bool, optional
            Whether to return the Hessian of the dual with respect to the lagrange multipliers. Default is False.
        penalty_vectors : list, optional
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
        grad_penalty, hess_penalty = [], []

        P_diag = self._add_projectors(lags)
        P = sp.diags_array(P_diag)
        A = self._get_total_A(P)
        S = self._get_total_S(P)

        Acho, xstar, grad_x = self._get_xstar(A, S, get_xgrad = get_hess) # grad_x is needed for calculating the hessian. xstar is sufficient for the gradient. 
        dualval = self.c0 + np.real(xstar.conjugate() @ A @ xstar) 
        
        if get_grad: # This is grad_lambda (not grad_x)
            A2_xstar = self.A2 @ xstar
            # First term: -Re(xstar.conj() @ self.A1 @ (self.Pdiags[:, i] * (self.A2 @ xstar)))
            # self.Pdiags has shape (N_diag, N_projectors), A2_xstar has shape (N_diag,)
            # We want to multiply each column of Pdiags elementwise with A2_xstar
            # Same method for term 2: 2*Re(xstar.conj() @ self.A2.T.conj() @ (self.Pdiags[:, i].conj() * self.s1))
            term1 = -np.real(xstar.conj() @ self.A1 @ (self.Pdiags * A2_xstar[:, np.newaxis]))  # Shape: (N_diag, N_projectors)
            term2 = 2 * np.real(A2_xstar.conj() @ (self.Pdiags.conj() * self.s1[:, np.newaxis]))  # Shape: (N_diag, N_projectors)
            grad = term1 + term2

            # Below is the for loop method for reference (slower)
            # grad = np.zeros(self.Pdiags.shape[1])
            # for i in range(len(grad)): 
            #     grad[i] = -np.real(xstar.conj() @ self.A1 @ (self.Pdiags[:, i] * (self.A2 @ xstar))) + 2*np.real(xstar.conj() @ self.A2.T.conj() @ (self.Pdiags[:, i].T.conj() * self.s1))

        # Boundary penalty for the PSD boundary 
        dualval_penalty = 0.0 
        if len(penalty_vectors) > 0:
            penalty_matrix = np.column_stack(penalty_vectors)
            A_inv_penalty = Acho.solve_A(penalty_matrix)
            dualval_penalty += np.sum(np.real(A_inv_penalty.conj() * penalty_matrix)) # multiplies columns with columns, sums all at once

            if get_grad: 
                grad_penalty = np.zeros(len(grad))
                for j in range(penalty_matrix.shape[1]):
                    grad_penalty += -np.real(A_inv_penalty[:, j].conj().T @ self.A1 @ (self.Pdiags * (self.A2 @ A_inv_penalty[:, j])[:, np.newaxis]))

                    # for loop method
                    # for i in range(len(grad)): 
                    #     grad_penalty[i] += -np.real(A_inv_penalty[:, j].conj().T @ self.A1 @ (self.Pdiags[:, i] * (self.A2 @ A_inv_penalty[:, j])))

        DualAux = namedtuple('DualAux', ['dualval_real', 'dualgrad_real', 'dualval_penalty', 'grad_penalty'])
        dual_aux = DualAux(dualval_real=dualval, dualgrad_real=grad, dualval_penalty=dualval_penalty, grad_penalty=grad_penalty)

        if len(penalty_vectors) > 0:
            return dualval + dualval_penalty, grad + grad_penalty, hess, dual_aux
        else:
            return dualval, grad, hess, dual_aux

    def _get_PSD_penalty(self, lags) -> np.ndarray:
        """
        Returns the eigenvector for the smallest eigenvalue. This vector can be used to calculate the PSD boundary of the dual function.
        In theory, other PSD penalties exist, such as the determinant. We found that this works well for photonic limits. 

        Parameters
        ----------
        lags : np.ndarray
            Lagrange multipliers from which A will be calculated

        Returns
        -------
        eigv : np.ndarray
            Eigenvector corresponding to the lowest eigenvalue of A(lags)
        """
        A = self._get_total_A(sp.diags_array(self._add_projectors(lags)))
        eigw, eigv = spla.eigsh(A, k=1, sigma=0.0, which='LM', return_eigenvectors=True)
        aux = eigw
        return eigv[:,0], aux
    
    def is_dual_feasible(self, lags: np.ndarray) -> bool:
        """
        Checks if a set of Lagrange multipliers is dual feasible by attempting a Cholesky decomposition.
        This is efficient because the sparsity pattern of the total A matrix is known and does not change.

        Arguments
        ---------
        lags: np.ndarray
            A 1-dimensional numpy array of Lagrange multipliers.

        Returns
        -------
        bool
            True if the total A matrix is positive semidefinite (dual feasible).
        """
        P_diag = self._add_projectors(lags)
        P = sp.diags_array(P_diag)
        A = self._get_total_A(P)
        try:
            Acho = self.Achofac.cholesky(A)
            tmp = Acho.L() # Have to access the factor for the decomposition to be actually done. 
            return True
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            return False

    def find_feasible_lags(self, start: float = 0.1, limit: float = 1e8) -> np.ndarray:
        """
        Find a feasible point for the dual problem. This assumes that for large enough lags[1], A is PSD.

        Parameters
        ----------
        start : float (optional, default 1.0)
            The start value for lags[1]
        limit : float (optional, default 1e8)
            A maximum value for lags[1] before the method gives up and raises an error.

        Returns
        -------
        init_lags : np.ndarray
            A set of Lagrange multipliers such that A is PSD.
        """
        init_lags = np.random.random(self.Pdiags.shape[1]) * 1e-6  # Start with small positive lags
        init_lags[1] = start
        while self.is_dual_feasible(init_lags) is False:
            init_lags[1] *= 1.5
            if init_lags[1] > limit:
                raise ValueError("Could not find a feasible point for the dual problem.")
            
        if self.verbose > 0: print(f"Found feasible point for dual problem: {init_lags} with dualvalue {self.get_dual(init_lags)[0]}")
        return init_lags

    def solve_current_dual_problem(self, method: str, opt_params: dict = None, init_lags: np.ndarray = None):
        """
        Solves the current dual problem using the specified method.

        Parameters
        ----------
        method : str
            The method to use for solving the dual problem. 'newton' or 'bfgs'
        """
        optfunc = self.get_dual
        feasibility_func = self.is_dual_feasible
        penalty_vector_func = self._get_PSD_penalty
        is_convex = True 
        if opt_params is None:
            opt_params = {'opttol': 1e-2, 'gradConverge': False, 'min_inner_iter': 5, 'max_restart': np.inf, 'penalty_ratio': 1e-2, 'penalty_reduction': 0.1, 'break_iter_period': 20, 'verbose': self.verbose-1, 'penalty_vector_list': []}

        if init_lags is None:
            init_lags = self.find_initial_feasible_point()

        if method == 'newton':
            raise NotImplementedError("SparseSharedProjQCQP cannot use Newton's method yet.")
        elif method == 'bfgs':
            optimizer = BFGS(optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params)
            lambda_opt, dualval_opt, grad_opt = optimizer.run(init_lags)
            self.current_dual, self.dual_lambda, self.current_grad, self.current_hess = dualval_opt, lambda_opt, grad_opt, []  # hess is not computed in BFGS
            return self.current_dual, self.dual_lambda, self.current_grad, self.current_hess
        else: 
            raise ValueError(f"Unknown method '{method}' for solving the dual problem. Use newton or bfgs.")

    def solve_primal_gurobi(self, gurobi_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """
        Solves the primal QCQP problem using Gurobi by calling an external solver function.
        This finds a global solution, and can be expensive, but can also quantify the duality gap. 
        Requires a Gurobi installation and license. 

        Parameters
        ----------
        gurobi_params : dict, optional
            Parameters to pass to Gurobi (e.g., {'NonConvex': 2, 'TimeLimit': 3600}).

        Returns
        -------
        x_optimal : np.ndarray
            The optimal complex vector x, or None if not found.
        obj_val : float
            The optimal objective value, or None if not found.
        status : str
            The Gurobi optimization status.
        
        Raises
        ------
        ImportError
            If the Gurobi solver module or GurobiPy itself cannot be imported.
        """
        # raise NotImplementedError("Gurobi solver is not implemented yet.")
    
        Warning("Solving the primal problem is expensive! Are you sure you want to call this function?")

        try:
            # Import locally to avoid circular dependencies at module load time
            # and to make Gurobi an optional dependency for qcqp.py
            from .gurobi_solver import solve_sparse_qcqp_primal_with_gurobi
        except ImportError as e:
            raise ImportError(
                "Could not import Gurobi solver components. "
                "Ensure GurobiPy is installed."
            ) from e
            
        return solve_sparse_qcqp_primal_with_gurobi(self, gurobi_params)
    
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