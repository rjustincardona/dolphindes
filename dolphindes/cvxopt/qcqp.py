"""
Dual Problem Interface

"""

__all__ = ['SparseSharedProjQCQP'] 

import copy
import numpy as np 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod 
from .optimization import BFGS, Alt_Newton_GD
from dolphindes.util import Sym
from collections import namedtuple
from typing import Optional, Dict, Any, Tuple # For type hinting the new method

class _SharedProjQCQP():
    """Represents a QCQP with a single type of constraint over projection regions. 
    Parent class, should not be instantiated directly as it is missing key functionality. 

    Problem is 
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j. 

    Attributes
    ----------
    A0 : LinearOperator
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
    Achofac : sksparse.cholmod.Cholesky | np.ndarray
        The Cholesky factorization of the total A matrix, which is updated when needed.
    current_dual : float
        The current dual solution, which is only updated when the dual problem is solved.
    current_lags : np.ndarray
        The current Lagrangian multipliers, which is only updated when the dual problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only updated when the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only updated when the dual problem is solved. 
    """
    def __init__(self, A0: np.ndarray | sp.csc_array, s0: np.ndarray, c0: float, A1: sp.csc_array, A2: sp.csc_array, 
                 s1: np.ndarray, Pdiags: np.ndarray, verbose: float = 0):
        self.A0 = A0
        self.s0 = s0
        self.s1 = s1
        self.c0 = c0
        self.A1 = sp.csc_array(A1)
        self.A2 = sp.csc_array(A2)
        self.verbose = verbose
        self.Pdiags = Pdiags
        self.Achofac = None
        self.current_dual = None 
        self.current_grad = None 
        self.current_hess = None
        self.current_lags = None 
        self.current_xstar = None 

    def __deepcopy__(self, memo):
        # custom __deepcopy__ because Achofac is not pickle-able
        new_QCQP = SparseSharedProjQCQP.__new__(SparseSharedProjQCQP)
        for name, value in self.__dict__.items():
            if name != 'Achofac':
                setattr(new_QCQP, name, copy.deepcopy(value, memo))
        
        new_QCQP._update_chofac()  # Recompute the Cholesky factorization. If dense, will use self.current_lags. 
        return new_QCQP
    
    def get_number_constraints(self) -> int:
        """Returns the number of constraints in the QCQP"""
        return self.Pdiags.shape[1]
    
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
        return self.Pdiags @ lags
    
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
    
class SparseSharedProjQCQP(_SharedProjQCQP):
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is 
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j. Representas 

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
    current_lags : np.ndarray
        The current Lagrangian multipliers, which is only updated when the dual problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only updated when the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only updated when the dual problem is solved. 
    """
    def __init__(self, A0: sp.csc_array, s0: np.ndarray, c0: float, A1: sp.csc_array, A2: sp.csc_array, 
                 s1: np.ndarray, Pdiags: np.ndarray, verbose: float = 0):
        super().__init__(A0, s0, c0, A1, A2, s1, Pdiags, verbose)
        self.A0 = sp.csc_array(A0) # Convert in case the user passes a dense array or other format 
        self.compute_precomputed_values()  # Precompute values for efficiency
        
    def __repr__(self):
        return f"SparseSharedProjQCQP of size {self.A0.shape[0]}^2 with {self.Pdiags.shape[1]} projectors."

    def compute_precomputed_values(self):
        # Precompute the A constraint matrices for each projector. This makes _get_total_A faster. 
        self.precomputed_As = []
        for i in range(self.Pdiags.shape[1]):
            Ak = Sym(self.A1 @ sp.diags_array(self.Pdiags[:, i], format='csr') @ self.A2)
            self.precomputed_As.append(Ak)
        
        print(f"Precomputed {self.Pdiags.shape[1]} A matrices for the projectors.")

        # (Fs)_k = A_2^dagger P_k^dagger s1
        self.Fs = self.A2.conj().T @ (self.Pdiags.conj().T * self.s1).T

        self._update_chofac()


    def _get_total_A(self, lags: np.ndarray) -> sp.csc_array:
        """Gets the total A matrix for the QCQP = A0 + sum_j lag[j] A1 P_j A2 given sum_j lag[j] P_j"""
        return self.A0 + sum(lags[i] * self.precomputed_As[i] for i in range(len(lags)))
    
    def _get_total_S(self, Pdiag: np.ndarray) -> np.ndarray:
        """Gets the total S vector for the QCQP = s0 + sum_j lag[j] P_j^dagger s1 given P = sum_j lag[j] P_j"""
        return self.s0 + self.A2.T.conj() @ (Pdiag.conj() * self.s1)

    def _update_chofac(self) -> sksparse.cholmod.Factor:
        """Updates the cholesky factorization of the total A
        
        Returns
        -------
        sksparse.cholmod.Cholesky
            The updated cholesky factorization of the total A matrix.
        """
        random_lags = np.random.rand(self.Pdiags.shape[1])
        # P = self._add_projectors(random_lags)
        A = self._get_total_A(random_lags)
        if self.verbose > 1: print(f"analyzing A of format and shape {type(A)}, {A.shape} and # of nonzero elements '{A.count_nonzero()}")
        self.Achofac = sksparse.cholmod.analyze(A)
        return self.Achofac

    def _get_xstar(self, lags: np.ndarray) -> tuple[sksparse.cholmod.Factor, np.ndarray, float]:
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
        xAx : float
            x_star.conj().T @ A @ x_star, the dual function value
        """

        P_diag = self._add_projectors(lags)
        # P = sp.diags_array(P_diag, format='csc')
        A = self._get_total_A(lags)
        S = self._get_total_S(P_diag)

        Acho = self.Achofac.cholesky(A)
        x_star = Acho.solve_A(S)
        xAx = np.real(x_star.conjugate() @ A @ x_star) 

        return Acho, x_star, xAx
        
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
            If True, grad is also automatically computed regardless of what get_grad specifies. 
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

        grad, hess = [], []
        grad_penalty, hess_penalty = [], []

        Acho, xstar, dualval = self._get_xstar(lags)
        dualval += self.c0

        if get_hess:
            if not hasattr(self, 'precomputed_As'):
                raise AttributeError('precomputed_As needed for computing Hessian')
                # this assumes that in the future we may consider making precomputed_As optional
                # can also compute the Hessian without precomputed_As, leave for future implementation if useful
                
            # useful intermediate computations
            # (Fx)_k = -Sym(A_1 P_k A2) x_star = -A_k @ x_star
            
            Fx = np.zeros((len(xstar),len(self.precomputed_As)), dtype=complex)
            for k,Ak in enumerate(self.precomputed_As):
                Fx[:,k] = -Ak @ xstar
            
            grad = np.real(xstar.conj() @ (Fx + 2*self.Fs)) #get_hess implies get_grad also
            
            Ftot = Fx + self.Fs
            hess = 2*np.real(Ftot.conj().T @ Acho.solve_A(Ftot))
                
        elif get_grad: # This is grad_lambda (not grad_x); elif since get_hess automatically computes grad
            # First term: -Re(xstar.conj() @ self.A1 @ (self.Pdiags[:, i] * (self.A2 @ xstar))). Second term: 2*Re(xstar.conj() @ self.A2.T.conj() @ (self.Pdiags[:, i].conj() * self.s1))
            # self.Pdiags has shape (N_diag, N_projectors), A2_xstar has shape (N_diag,)
            # We want to multiply each column of Pdiags elementwise with A2_xstar. 
            # However, we know that sum_i w_i A_ij v_i = sum_i (w_i * v_i) A_ij. LHS is expression right below, RHS is below so we avoid dense intermediate matrices. 
            A2_xstar = self.A2 @ xstar      # Shape: (N_p,)
            x_conj_A1 = xstar.conj() @ self.A1  # Shape: (N_p,) where N_p = self.A1.shape[1]
            # term1 = -np.real((xstar.conj() @ self.A1) @ (self.Pdiags * A2_xstar[:, np.newaxis]))  # Shape: (N_diag, N_projectors)
            term1 = -np.real( (x_conj_A1 * A2_xstar) @ self.Pdiags )

            # term2 = 2 * np.real(A2_xstar.conj() @ (self.Pdiags.conj() * self.s1[:, np.newaxis])) #. Same as above. 
            term2 = 2 * np.real( (A2_xstar.conj() * self.s1) @ self.Pdiags.conj() )
            grad = term1 + term2


        # Boundary penalty for the PSD boundary 
        dualval_penalty = 0.0 
        if len(penalty_vectors) > 0:
            penalty_matrix = np.column_stack(penalty_vectors)
            A_inv_penalty = Acho.solve_A(penalty_matrix)
            dualval_penalty += np.sum(np.real(A_inv_penalty.conj() * penalty_matrix)) # multiplies columns with columns, sums all at once

            if get_hess:
                # get_hess implies get_grad also
                grad_penalty = np.zeros(len(grad))
                hess_penalty = np.zeros((len(grad),len(grad)))
                Fv = np.zeros((penalty_matrix.shape[0],len(grad)), dtype=complex)
                for j in range(penalty_matrix.shape[1]):
                    for k,Ak in enumerate(self.precomputed_As):
                        # yes this is a double for loop, hessian for fake sources is likely a speed bottleneck
                        Fv[:,k] = Ak @ A_inv_penalty[:,j]
                    
                    grad_penalty += np.real(-A_inv_penalty[:,j].conj().T @ Fv)
                    hess_penalty += 2*np.real(Fv.conj().T @ Acho.solve_A(Fv))
                    
            elif get_grad: 
                grad_penalty = np.zeros(len(grad))
                for j in range(penalty_matrix.shape[1]):
                    # slow: for i in range(len(grad)): grad_penalty[i] += -np.real(A_inv_penalty[:, j].conj().T @ self.A1 @ (self.Pdiags[:, i] * (self.A2 @ A_inv_penalty[:, j]))) # for loop method
                    # fast: grad_penalty += -np.real((A_inv_penalty[:, j].conj().T @ self.A1) @ (self.Pdiags * (self.A2 @ A_inv_penalty[:, j])[:, np.newaxis]))

                    # Same as above (fastest)
                    A_inv_penalty_j_A1 = A_inv_penalty[:, j].conj().T @ self.A1  
                    A2_A_inv_penalty_j = self.A2 @ A_inv_penalty[:, j]  # Shape: (N_p,)
                    grad_penalty += -np.real( (A_inv_penalty_j_A1 * A2_A_inv_penalty_j) @ self.Pdiags ) 

        DualAux = namedtuple('DualAux', ['dualval_real', 'dualgrad_real', 'dualval_penalty', 'grad_penalty'])
        dual_aux = DualAux(dualval_real=dualval, dualgrad_real=grad, dualval_penalty=dualval_penalty, grad_penalty=grad_penalty)

        if len(penalty_vectors) > 0:
            return dualval + dualval_penalty, grad + grad_penalty, hess + hess_penalty, dual_aux
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
        A = self._get_total_A(lags)
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
        A = self._get_total_A(lags)
        try:
            Acho = self.Achofac.cholesky(A)
            tmp = Acho.L() # Have to access the factor for the decomposition to be actually done. 
            return True
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            return False

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
            init_lags = self.find_feasible_lags()

        if method == 'newton':
            optimizer = Alt_Newton_GD(optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params)
        elif method == 'bfgs':
            optimizer = BFGS(optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params)
        else: 
            raise ValueError(f"Unknown method '{method}' for solving the dual problem. Use newton or bfgs.")
        
        self.current_lags, self.current_dual, self.current_grad = optimizer.run(init_lags)
        _, self.current_xstar, _ = self._get_xstar(self.current_lags)

        return self.current_dual, self.current_lags, self.current_grad, self.current_hess, self.current_xstar
        
    
    def refine_projectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Doubles the number of projectors, refining the number of constraints to smaller regions. Multipliers will be selected so that dual value remains constant and can be further optimized from existing point. 

        For each projector P_j (columns of Pdiags) that doesn't project to a single pixel, split it into two projectors P1 and P2 such that P_j = P1 + P2 with half (or near half) the nonzero entries; j>2 (zero-th and first projector is always left as is). If there are only two projectors, keep both split projectors and original ones. 

        Then, form a new Pdiags with the new projectors. Furthermore, extend lags. For each split projector P_j -> P_j, P_j+1, make lags[j] = lags[j], lags[j+1] = lags[j]. 

        Updates Pdiags and current_lags attributes. Verifies the dual value remains the same after refinement.

        Arguments
        ---------
        None

        Returns
        -------
        self.Pdiags, self.current_lags
        """
        assert self.current_lags is not None, "Cannot refine projectors until an existing problem is solved. Run solve_current_dual_problem first."
        assert np.all(self.Pdiags[:, 0] == 1), "The zeroth projector must contain all ones (identity)"
        assert np.all(np.isclose(self.Pdiags[:, 1], -1j)), "The second projector must contain all -1j values (-1j * identity)"

        new_Pdiags_cols = []
        new_lags_list = []

        # Handle the first projectors (j=0,1) - always kept as is
        new_Pdiags_cols.append(self.Pdiags[:, 0])
        new_lags_list.append(self.current_lags[0])

        new_Pdiags_cols.append(self.Pdiags[:, 1])
        new_lags_list.append(self.current_lags[1])

        # Iterate through the rest of the projectors (j > 0)
        split_limit = 0 if self.Pdiags.shape[1] == 2 else 2
        for j in range(split_limit, self.Pdiags.shape[1]):
            P_j_diag = self.Pdiags[:, j]
            current_lag_j = self.current_lags[j]

            # Find non-zero elements (pixels) in the current projector
            # Assuming projector diagonals are binary (0 or 1)
            nonzero_indices = np.where(P_j_diag != 0)[0]
            num_nonzero = len(nonzero_indices)

            if num_nonzero == 1: # Projector is already a single pixel, keep it as is
                new_Pdiags_cols.append(P_j_diag)
                #new_lags_list.append(current_lag_j)
            elif num_nonzero > 1: # Split the projector
                split_point = num_nonzero // 2
                indices1 = nonzero_indices[:split_point]
                indices2 = nonzero_indices[split_point:]
                # Create new projector P1
                P1_diag_new = np.zeros_like(P_j_diag)
                P1_diag_new[indices1] = P_j_diag[indices1]  # Copy values from original projector
                
                # Create new projector P2
                P2_diag_new = np.zeros_like(P_j_diag)
                P2_diag_new[indices2] = P_j_diag[indices2]  # Copy values from original projector

                if not np.all(P1_diag_new[1:] == 0): new_Pdiags_cols.append(P1_diag_new)
                if not np.all(P2_diag_new[1:] == 0): new_Pdiags_cols.append(P2_diag_new)
                
                if self.verbose > 1:
                    print(f"Split projector {j} (lag: {current_lag_j:.2e}) with {num_nonzero} non-zeros into two projectors with {len(indices1)} and {len(indices2)} non-zeros.")

            else:
                raise ValueError(f"Unexpected number of non-zero elements in projector {j}: {num_nonzero}. Projector should have at least one non-zero element.")

        # Update Pdiags and current_lags
        # We need the overall P (and therefore the dual value) to stay constant upon projector refinement. 
        # Here, we solve the system of equations to do that. It should always be solvable. 
        new_Pdiags = np.column_stack(new_Pdiags_cols)
        new_Pdiags_real = np.vstack([np.real(new_Pdiags), np.imag(new_Pdiags)])  # Convert complex to real for least squares
        old_Pdiags_real = np.vstack([np.real(self.Pdiags), np.imag(self.Pdiags)])  # Convert complex to real for least squares
        self.current_lags, residuals, rank, s = np.linalg.lstsq(new_Pdiags_real, old_Pdiags_real @ self.current_lags, rcond=None)
        self.Pdiags = new_Pdiags
        
        # Re-calculate precomputed_As as the projectors have changed
        self.compute_precomputed_values()
    
        # Reset current dual, grad, hess, xstar as they are for the old problem structure
        new_dual, new_grad, new_hess, dual_aux = self.get_dual(self.current_lags, get_grad=True, get_hess=False)
        if self.verbose >= 1: print(f'previous dual: {self.current_dual}, new dual: {new_dual} (should be the same)')
        assert np.isclose(new_dual, self.current_dual, rtol=1e-2), "Dual value should be the same after refinement."

        self.current_dual = new_dual
        self.current_grad = new_grad
        self.current_hess = new_hess
        self.current_xstar = self.current_xstar
        
        return self.Pdiags, self.current_lags

    def iterative_splitting_step(self, method : str = 'bfgs', max_cstrt_num : int = np.inf):
        """
        Iterative splitting step generator function that continues until pixel-level constraints are reached.

        Each call:
            1. Doubles the number of projectors using refine_projectors
            2. Solves the new dual problem and yields the results 
            3. Continues until all projectors are "one-hot" matrices (pixel-level constraints)

        Parameters
        ----------
        method : str
            Optimizer used for solving each iterative splitting step
        max_cstrt_num : int or np.inf
            termination condition based on maximum number of constraints.
            If np.inf, defaults to pixel level constraints
        Yields
        ------
        tuple
            Result of solve_current_dual_problem: 
            (current_dual, current_lags, current_grad, current_hess, current_xstar)
        """
        max_cstrt_num = min(max_cstrt_num, 2 * self.Pdiags.shape[0])
        # Check if we're already at termination condition
        if self.Pdiags.shape[1] >= max_cstrt_num:
            if self.verbose > 0:
                print("Projector number already above specified max or pixel level.")
            return

        # Continue splitting until number of constraints exceeds or equals max_cstrt_num
        while self.Pdiags.shape[1] < max_cstrt_num:
            if self.verbose > 0:
                if self.Pdiags.shape[1] == 2:
                    print(f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1]}")
                else: 
                    print(f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1] - 2}")
                
            # Refine projectors to get finer constraints
            self.refine_projectors()
            
            # Solve the dual problem with the new projectors
            result = self.solve_current_dual_problem(method, init_lags=self.current_lags)
            
            # Yield the result to the caller
            yield result

            # Check if we've reached pixel level after this iteration
            if self.Pdiags.shape[1] >= max_cstrt_num:
                if self.verbose > 0:
                    print("Reached max number of projectors or pixel-level projectors. Refinement complete.")
                break

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
        raise NotImplementedError("Gurobi solver is not implemented yet.")
    
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
    pass 