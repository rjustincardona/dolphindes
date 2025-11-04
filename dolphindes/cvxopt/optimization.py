"""
Optimizers

"""

__all__ = ["BFGS", "Alt_Newton_GD"]

import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.interpolate import AAA

def bisect(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return bisect(f, a, b+abs(b-a), tol=tol, max_iter=max_iter)
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    raise ValueError("Maximum iterations reached without convergence")

def pole_aaa(f, x0, tol=1e-4, r=1e-1, max_iter=5, max_restart=7):
    xs = [x0]
    fs = [f(x0)]
    for _ in range(max_restart):
        idx = np.argmin(np.abs(fs))
        xs = [xs[idx]]
        fs = [fs[idx]]
        xs.append(x0 - np.sign(fs[-1]) * r)
        fs.append(f(xs[-1]))
        for _ in range(max_iter):
            a = AAA(xs, fs)
            root = np.max(np.real(a.roots()))
            xs.append(root)
            fs.append(f(xs[-1]))
            if np.abs(fs[-1]) < tol:
                return np.real(np.max(a.poles())), xs[-1]
        # print(xs, fs)
        # x_plot = np.sort(list(np.linspace(np.min(xs), np.max(xs))) + xs)
        # y_plot = [a(x) for x in x_plot]
        # f_plot = [f(x) for x in x_plot]
        # plt.scatter(xs, fs, color='orange', label='Sampled Points')
        # plt.plot(x_plot, y_plot, label='AAA Approximation')
        # plt.plot(x_plot, f_plot, label='f(x)')
        # ymax = max(np.max(y_plot), np.max(f_plot))
        # ymin = min(np.min(y_plot), np.min(f_plot))
        # plt.vlines(xs[-1], ymin=ymin, ymax=ymax, colors='r', linestyles='dashed', label='Estimated Root')
        # plt.vlines(xs[0], ymin=ymin, ymax=ymax, colors='g', linestyles='dashed', label='Initial Point')
        # plt.legend()
        # plt.show()
        # plt.close()
        
    raise ValueError("root_aaa did not converge")
                

class _Optimizer():
    """Base class for optimization algorithms.
    This abstract class provides a foundation for implementing various optimization
    algorithms. It manages optimization parameters, tracks optimization results, and 
    defines the interface for optimization algorithms.
    
    Arguments
    ---------
        optfunc: Callable function to be optimized.
        valid_func: Callable function that checks if a solution is valid.
        penalty_vector_func: Callable function that returns penalty vectors.
        is_convex: bool 
            Boolean indicating if the optimization problem is convex.
        opt_params: Dictionary containing optimization parameters. These parameters 
            override the default parameters defined in OPT_PARAMS_DEFAULTS
    
    Attributes
    ----------
        optfunc: The optimization objective function, returns a tuple (f(x), grad_f(x), hess_f(x)), which may contain
            zeroes if the method does not need them
        feasible_func: Function to check if a solution is feasible
        penalty_vector_func: Function to compute penalty vectors given a point x 
        opt_params: Dictionary of optimization parameters.
        last_opt_x: The last optimized parameter vector.
        last_opt_fx: The function value at the last optimized point.
    
    Notes
    -----
        Subclasses must implement the `run` method to define the specific 
        optimization algorithm. Use the `get_last_opt` method to retrieve
        the results of the most recent optimization.
    """

    OPT_PARAMS_DEFAULTS = {'opttol': 1e-8, 'gradConverge': False, 'min_inner_iter': 5, 'max_restart': np.inf, 'penalty_ratio': 1e-2, 'penalty_reduction': 0.1, 'break_iter_period': 50, 'verbose': 0}

    def __init__(self, optfunc, _get_xstar, A1, A2, s1, Pdiags, aU, _get_total_A, feasible_func, penalty_vector_func, is_convex, opt_params):
        self.optfunc = optfunc
        self.feasible_func = feasible_func
        self.penalty_vector_func = penalty_vector_func
        
        self.is_convex = is_convex
        self.opt_params = {**self.OPT_PARAMS_DEFAULTS, **opt_params}
        self.penalty_vector_list = []

        if self.opt_params['verbose'] > 0:
            print('Optimizer initialized with parameters:')
            for k, v in self.opt_params.items():
                print(f'{k}: {v}')

        self.opt_x = None 
        self.opt_fx = None
        self.verbose = self.opt_params['verbose']

        # Partial Dual stuff
        self._get_xstar = _get_xstar
        self.A1 = A1
        self.A2 = A2
        self.s1 = s1
        self.Pdiags = Pdiags
        self._get_total_A = _get_total_A
        self.aU = aU

    def C(self, x0, z, penalty_vectors=[]):
        lags = x0.copy()
        lags[1] = z
        xstar, _ = self._get_xstar(lags)
        A2_xstar = self.A2 @ xstar
        x_conj_A1 = xstar.conj() @ self.A1
        result = 2 * np.real( (A2_xstar.conj() * self.s1) @ self.Pdiags.conj()[:, 1] ) - np.real( (x_conj_A1 * A2_xstar) @ self.Pdiags[:, 1] )
        
        if len(penalty_vectors) > 0:
            penalty_matrix = np.column_stack(penalty_vectors)
            A_inv_penalty = spla.spsolve(self._get_total_A(lags), penalty_matrix)
            grad_penalty = 0
            for j in range(penalty_matrix.shape[1]):
                A_inv_penalty = np.reshape(A_inv_penalty, penalty_matrix.shape)
                A_inv_penalty_j_A1 = A_inv_penalty[:, j].conj().T @ self.A1  
                A2_A_inv_penalty_j = self.A2 @ A_inv_penalty[:, j]  # Shape: (N_p,)
                grad_penalty += -np.real( (A_inv_penalty_j_A1 * A2_A_inv_penalty_j) @ self.Pdiags[1] )
        else:
            grad_penalty = 0
        return result + grad_penalty
    
    def min_g(self, x0):
        lags = x0.copy()
        lags[1] = 0
        Lq = self._get_total_A(lags)
        pole, vectors = spla.eigs(Lq, M=self.aU, k=1, return_eigenvectors=True)
        pole = pole[0]
        if pole > 0:
            pole, vectors = spla.eigs(Lq, M=self.aU, sigma=-pole, k=1, return_eigenvectors=True)
            pole = pole[0]
        pole = -pole.real

        try:
            _, root = pole_aaa(lambda z: self.C(x0, z), pole+1e-3)
            # print("aaa: ", root)
        except ValueError:
            root = bisect(lambda z: self.C(x0, z), a=pole+1e-4, b=pole+10)
            # print("bisect: ", root)
        # x_plot = np.linspace(pole-10, root+10, 1000)
        # y_plot = [self.C(x0, z) for z in x_plot]
        # plt.plot(x_plot, y_plot)
        # plt.vlines(pole, ymin=min(y_plot), ymax=max(y_plot), colors='r', linestyles='dashed', label='PSD boundary')
        # plt.vlines(root, ymin=min(y_plot), ymax=max(y_plot), colors='g', linestyles='dashed', label=r'Optimal $\gamma$')
        # plt.legend()
        # plt.xlabel(r"$\gamma$")
        # plt.ylabel("$C(\gamma)$")
        # plt.yscale('asinh')
        # plt.show()
        # plt.close()
        return pole, root



    def get_last_opt(self):
        """Get the last optimized x and f(x) values."""
        return self.opt_x, self.opt_fx
    
    def _line_search(self, dir : np.ndarray, x0 : np.ndarray, fx0 : np.ndarray, grad : np.ndarray, init_step_size : float) -> float:
        """This method implements a two-phase backtracking line search:
        1. First finds a feasible step size by backtracking until the point is feasible
        2. Then finds an optimal step size satisfying the Armijo condition while minimizing the function value
        
        Parameters
        ----------
        dir : numpy.ndarray
            The search direction vector, normalized to 1
        x0 : numpy.ndarray
            The starting point for the line search
        fx0 : float
            Function value at the starting point, f(x0)
        grad : numpy.ndarray
            Gradient vector at the starting point
        init_step_size : float
            Initial step size to begin the line search
        add_penalty : bool, optional
            Flag to add penalty term to the line search, by default False
        
        Returns
        -------
        float
            The optimal step size found by the line search
        
        Notes
        -----
        The Armijo condition ensures sufficient decrease in function value:
            f(x + α·d) ≤ f(x) + c·α·∇f(x)ᵀd
        where c is a small constant (c_A = 1e-4 in this implementation)"
        """

        # First, find feasible alpha 
        c_reduct = 0.7
        alpha = alpha_start = init_step_size

        if self.verbose >= 3: print(f"\nStarting line search with parameters alpha_start = {alpha_start}, alpha = {alpha}")
        back_iter = 0
        # while not self.feasible_func(x0 + alpha * dir):
        pole, root = self.min_g(x0 + alpha * dir)
        while pole > x0[1] + alpha * dir[1]+1e-3:
            back_iter += 1
            if alpha < 1e-5:
                break
            alpha *= c_reduct
            pole, root = self.min_g(x0 + alpha * dir)
            # try:
            #     pole, root = pole_aaa(lambda z: self.C(x0 + alpha * dir, z), x0=root)
            # except ValueError:
            #     pole, root = self.min_g(x0 + alpha * dir)
        alpha_opt = alpha
        alpha_feas = alpha


        # Next, find optimal alpha 
        c_A = 1e-4
        opt_val = np.inf
        grad_direction = dir @ grad
        while True:
            tmp_value, _, _, _ = self.optfunc(x0 + alpha*dir, get_grad=False, get_hess=False, penalty_vectors=self.penalty_vector_list)
            if self.verbose>=3:
                print('backtracking tmp_value', tmp_value)
            if tmp_value < opt_val: #the dual is still decreasing as we backtrack, continue
                opt_val = tmp_value; alpha_opt=alpha
            else:
                break
            if not self.is_convex and tmp_value <= fx0 + c_A * alpha * grad_direction: #Armijo backtracking condition if problem is not convex
                alpha_opt = alpha
                break
            
            alpha *= c_reduct
        
        if self.verbose >= 2: print(f"Line search found optimal step size: {alpha_opt}")
        pole, root = self.min_g(x0 + alpha * dir)
        # calculate second derivative of C at root
        h = (pole - root) / 2
        C_plus = self.C(x0 + alpha * dir, root + h)
        C_minus = self.C(x0 + alpha * dir, root - h)
        C_root = self.C(x0 + alpha * dir, root)
        C_dd = (C_plus - 2 * C_root + C_minus) / (h**2)

        return alpha_opt, alpha_feas, pole, root, C_dd
    
    def run(self, x0, tol=None):
        """Run the optimization routine with initial point x0 and tolerance tol

        Arguments
        ---------
            x0: ndarray
                Initial point for optimization
            tol: float
                Tolerance for optimization convergence

        Returns
        -------
            x_opt: ndarray
                Optimal point found by the optimizer
            x_grad: ndarray
                Gradient of the objective function at the optimal point
            f_opt: float
                Function value at the optimal point
        """
        raise NotImplementedError("Optimizer.run() must be implemented in subclasses")

class BFGS(_Optimizer):
    """Subclass of `Optimizer`, inherits its behavior.

    Additional Features:
    ---------------------
        - run() is implemented via BFGS optimization algorithm
        - _break_condition() is implemented to check for convergence
        - _update_hess_inv() is implemented to update the inverse Hessian approximation as part of BFGS
    """
    def __init__(self, *params):
        super().__init__(*params)
    
    def _break_condition(self, iter_num, iter_type):
        if iter_type == 'inner':
            # Directional stationarity residual convergence
            if iter_num > self.opt_params['min_inner_iter']:
                function_value = self.opt_fx 
                opttol = self.opt_params['opttol']
                fminus_xxgrad = function_value - np.dot(self.opt_x, self.xgrad)
                remaining_descent = np.abs(self.opt_x) @ np.abs(self.xgrad)
                gradConverge = self.opt_params['gradConverge']
                
                if gradConverge  and np.abs(function_value - fminus_xxgrad)< opttol * np.abs(fminus_xxgrad) and np.abs(remaining_descent)< opttol * np.abs(fminus_xxgrad) and np.linalg.norm(self.xgrad) < opttol * np.abs(function_value): 
                    return True 

                elif (not gradConverge) and np.abs(function_value-fminus_xxgrad) < opttol*np.abs(fminus_xxgrad) and np.abs(remaining_descent)<opttol*np.abs(fminus_xxgrad):
                    return True 
            
            # Simple objective value convergence 
            if iter_num % self.opt_params['break_iter_period'] == 0:
                if self.verbose > 0 :print(f"iter_num: {iter_num}, prev_fx: {self.prev_fx}, opt_fx: {self.opt_fx}, opttol: {self.opt_params['opttol']}")
                if np.abs(self.prev_fx - self.opt_fx) < np.abs(self.opt_fx)*self.opt_params['opttol'] or np.isclose(self.opt_fx, 0, atol=1e-14):
                    return True
                self.prev_fx = self.opt_fx

        elif iter_type == 'outer':
            # Outer objective value convergence
            if np.abs(self.prev_fx_outer - self.opt_fx) < np.abs(self.opt_fx)*self.opt_params['opttol'] or np.isclose(self.opt_fx, 0, atol=1e-14):
                return True
            
            # If a max number of outer iterations was specified, check for that 
            if iter_num > self.opt_params['max_restart']:
                if self.verbose >= 2: print(f"Maximum number of outer iterations reached: {self.opt_params['max_restart']}")
                return True
        
        return False
    
    def _update_Hinv(self, Hinv, new_grad, old_grad, delta, reset=False):
        """Update the inverse Hessian approximation using the BFGS formula.
        
        The BFGS (Broyden-Fletcher-Goldfarb-Shanno) update is a quasi-Newton method 
        that approximates the inverse Hessian matrix based on gradient information.
        This method builds up curvature information as optimization progresses to
        help inform the direction of the next step.
        
        Parameters
        ----------
        Hinv : numpy.ndarray
            Current inverse Hessian approximation
        new_grad : numpy.ndarray
            Gradient at the new point
        old_grad : numpy.ndarray
            Gradient at the previous point
        delta : numpy.ndarray
            Step taken to reach the new point (x_new - x_old)
        reset : bool, optional
            If True, reset the inverse Hessian to identity, by default False
            
        Returns
        -------
        numpy.ndarray
            Updated inverse Hessian approximation
            
        Notes
        -----
        The standard BFGS update formula is:
            H_{k+1} = (I - ρ*s_k*y_k^T)*H_k*(I - ρ*y_k*s_k^T) + ρ*s_k*s_k^T
        where:
            s_k = delta (step)
            y_k = new_grad - old_grad (change in gradient)
            ρ = 1/(y_k^T*s_k)
            
        The update is skipped if y_k^T*s_k is too small to avoid numerical instability.
        For quadratic functions, the inverse Hessian converges to the true inverse.
        """
        
        # The standard BFGS update formula for the inverse Hessian approximation
        # Note: delta is the step (s_k) and gamma is the change in gradient (y_k)
        
        if reset:
            return np.eye(Hinv.shape[0])  # Reset to identity if requested
        
        gamma = new_grad - old_grad  # Change in gradient
        gamma_dot_delta = gamma @ delta  # y_k^T * s_k
        
        # Skip update if gamma_dot_delta is too small (avoid division by zero or numerical instability)
        # if abs(gamma_dot_delta) < 1e-10:
        #     if self.verbose >= 3:
        #         print("Skipping Hinv update due to small gamma_dot_delta")
        #     return Hinv
            
        # Standard BFGS update formula
        rho = 1.0 / gamma_dot_delta
        I = np.eye(Hinv.shape[0])
        term1 = I - rho * np.outer(delta, gamma)
        term2 = I - rho * np.outer(gamma, delta)
        term3 = rho * np.outer(delta, delta)
        
        new_Hinv = term1 @ Hinv @ term2 + term3
            
        return new_Hinv
    
    def _add_penalty(self, opt_step_size, last_step_size, feas_step_size, x0, dir, opt_fx0, pole, root, curvature, e ):
        if np.isclose(opt_step_size, last_step_size, atol=0.0):
            # if no backtracking happened, can start with a more aggressive stepsize
            return opt_step_size * 2, False
        else:
            if (feas_step_size < last_step_size and np.isclose(opt_step_size, feas_step_size, atol=0.0)) or pole - root < 1e-2 or curvature > -1e-3 or e < 1e-1:
                # all the backtracking due to feasibility reasons, add penalty
                if self.verbose >= 2: print("Adding penalty due to feasibility wall.")
                penalty_vector, _ = self.penalty_vector_func(x0 + opt_step_size * dir)
                penalty_value = self.optfunc(x0, get_grad=False, get_hess=False, penalty_vectors=[penalty_vector])[0] 
                epsS = np.sqrt(self.opt_params['penalty_ratio']*np.abs(opt_fx0 / penalty_value))
                self.penalty_vector_list.append(epsS * penalty_vector)

                return opt_step_size, True
            return opt_step_size, False

    def run(self, x0, tol=None):
        self.opt_x = x0
        self.ndof = self.opt_x.size
        self.xgrad = np.zeros(self.ndof)
        self.prev_fx = np.inf
        self.prev_fx_outer = np.inf

        outer_iter_count = 0

        if self.verbose > 0:
            print(f"Starting optimization with x0 = {self.opt_x}")

        while True: # outer loop - penalty reduction
            self.penalty_vector_list = [] # reset penalty vectors
            self.opt_fx, self.xgrad, _, __ = self.optfunc(self.opt_x, get_grad=True, get_hess=False)

            last_step_size = 1.0 
            Hinv = np.eye(self.ndof)

            inner_iter_count = 0 
            
            if self.verbose > 0:
                print(f"Outer iteration {outer_iter_count}, penalty_ratio = {self.opt_params['penalty_ratio']}, opt_fx = {self.opt_fx}")

            while True:
                inner_iter_count += 1

                if self.verbose > 1:
                    print(f"Inner iteration {inner_iter_count}, opt_fx = {self.opt_fx}")

                BFGS_dir = - Hinv @ self.xgrad
                nBFGS_dir = BFGS_dir / np.linalg.norm(BFGS_dir)

                opt_step_size, feas_step_size, pole, root, curvature = self._line_search(nBFGS_dir, self.opt_x, self.opt_fx, self.xgrad, last_step_size) # Perform line search for step size 


                e = np.min(np.linalg.eigvals(Hinv))
                last_step_size, added_penalty = self._add_penalty(opt_step_size, last_step_size, feas_step_size, self.opt_x, nBFGS_dir, self.opt_fx, pole, root, curvature, e)

                delta = opt_step_size * nBFGS_dir
                old_grad = self.xgrad.copy()  # Save old gradient before update
                self.opt_x += delta

                new_opt_fx, new_grad, _, __ = self.optfunc(self.opt_x, get_grad=True, get_hess=False, penalty_vectors=self.penalty_vector_list) # Get new function value and gradient

                Hinv = self._update_Hinv(Hinv, new_grad, old_grad, delta, reset=added_penalty) # Update Hessian inverse. If added_penalty, will reset.

                self.opt_fx = new_opt_fx
                self.xgrad = new_grad

                if self._break_condition(inner_iter_count, 'inner'): break 
                # if inner_iter_count > 50: break

            if self._break_condition(outer_iter_count, 'outer'): break 
            self.prev_fx_outer = self.opt_fx

            outer_iter_count += 1
            self.opt_params['penalty_ratio'] *= self.opt_params['penalty_reduction']
            # if outer_iter_count > 3: break
        
        # pole, root = self.min_g(self.opt_x)
        # x_plot = np.linspace(pole-1e-3, max(self.opt_x[1], root), 100)
        # y_plot = [self.C(self.opt_x, z) for z in x_plot]
        # plt.plot(x_plot, y_plot, label='C(x0, z)')
        # plt.vlines(self.opt_x[1], ymin=min(y_plot), ymax=max(y_plot), colors='r', linestyles='dashed', label='Current x0[1]')
        # plt.vlines(root, ymin=min(y_plot), ymax=max(y_plot), colors='g', linestyles='dashed', label='Root z')
        # plt.vlines(pole, ymin=min(y_plot), ymax=max(y_plot), colors='orange', linestyles='dashed', label='Pole')
        # plt.legend()
        # plt.show()
        # plt.close()
            
        return self.opt_x, self.opt_fx, self.xgrad, None


class Alt_Newton_GD(_Optimizer):
    """
    Subclass of `Optimizer`, inherits its behavior.

    Additional Features:
    ---------------------
        - run() is implemented via alternating Newton and gradient descent steps; alternating improves stability
        - _break_condition() is implemented to check for convergence
    """
    
    def __init__(self, *params):
        super().__init__(*params)
    
    def _break_condition(self, iter_num, iter_type):
        if iter_type == 'inner':
            # Directional stationarity residual convergence
            if iter_num > self.opt_params['min_inner_iter']:
                function_value = self.opt_fx 
                opttol = self.opt_params['opttol']
                fminus_xxgrad = function_value - np.dot(self.opt_x, self.xgrad)
                remaining_descent = np.abs(self.opt_x) @ np.abs(self.xgrad)
                gradConverge = self.opt_params['gradConverge']
                
                if self.verbose>=3:
                    print(f"opt_fx: {self.opt_fx}, fminus_xxgrad: {fminus_xxgrad}, grad norm: {np.linalg.norm(self.xgrad)}")
                    
                if gradConverge  and np.abs(function_value - fminus_xxgrad)< opttol * np.abs(fminus_xxgrad) and np.abs(remaining_descent)< opttol * np.abs(fminus_xxgrad) and np.linalg.norm(self.xgrad) < opttol * np.abs(function_value): 
                    return True 

                elif (not gradConverge) and np.abs(function_value-fminus_xxgrad) < opttol*np.abs(fminus_xxgrad) and np.abs(remaining_descent)<opttol*np.abs(fminus_xxgrad):
                    return True 
                
            # Simple objective value convergence 
            if iter_num % self.opt_params['break_iter_period'] == 0:
                if self.verbose > 1 :print(f"iter_num: {iter_num}, prev_fx: {self.prev_fx}, opt_fx: {self.opt_fx}, opttol: {self.opt_params['opttol']}")
                if np.abs(self.prev_fx - self.opt_fx) < np.abs(self.opt_fx)*self.opt_params['opttol'] or np.isclose(self.opt_fx, 0, atol=1e-14):
                    return True
                self.prev_fx = self.opt_fx

        elif iter_type == 'outer':
            # Outer objective value convergence
            if np.abs(self.prev_fx_outer - self.opt_fx) < np.abs(self.opt_fx)*self.opt_params['opttol'] or np.isclose(self.opt_fx, 0, atol=1e-14):
                return True
            self.prev_fx_outer = self.opt_fx
            
            # If a max number of outer iterations was specified, check for that 
            if iter_num > self.opt_params['max_restart']:
                if self.verbose >= 1: print(f"Maximum number of outer iterations reached: {self.opt_params['max_restart']}")
                return True
        
        return False
    
    def _update_stepsize_add_penalty(self, opt_step_size, last_step_size, feas_step_size, x0, xdir, opt_fx0):
        if self.verbose >= 3:
            print(f"last_step_size: {last_step_size}, feas_step_size: {feas_step_size}, opt_step_size: {opt_step_size}")
        
        if np.isclose(opt_step_size, last_step_size, atol=0.0):
            # if no backtracking happened, can start with a more aggressive stepsize
            return opt_step_size * 2

        if feas_step_size < last_step_size and np.isclose(opt_step_size, feas_step_size, atol=0.0):
            # all the backtracking due to feasibility reasons, add penalty
            if self.verbose >= 2: print("Adding penalty due to feasibility wall.")
            penalty_vector, _ = self.penalty_vector_func(x0 + opt_step_size * xdir)
            penalty_value = self.optfunc(x0, get_grad=False, get_hess=False, penalty_vectors=[penalty_vector])[0] 
            epsS = np.sqrt(self.opt_params['penalty_ratio']*np.abs(opt_fx0 / penalty_value))
            self.penalty_vector_list.append(epsS * penalty_vector)

        return opt_step_size

    
    def run(self, x0):
        self.opt_x = x0
        self.ndof = x0.size
        self.xgrad = np.zeros(self.ndof)
        self.xhess = np.zeros((self.ndof,self.ndof))
        self.prev_fx = np.inf
        self.prev_fx_outer = np.inf
        
        outer_iter_count = 1

        if self.verbose > 0:
            print(f"Starting optimization with x0 = {self.opt_x}")

        while True: # outer loop - penalty reduction
            self.penalty_vector_list = [] # reset penalty vectors
            last_N_step_size = last_GD_step_size = 1.0 # reset step sizes
            
            inner_iter_count = 1
            
            if self.verbose > 0:
                print(f"Outer iteration {outer_iter_count}, penalty_ratio = {self.opt_params['penalty_ratio']}, opt_fx = {self.opt_fx}")
            
            while True:
                
                doN = (inner_iter_count % 2 == 1) # alternate between Newton and GD steps
                self.opt_fx, self.xgrad, self.xhess, _ = self.optfunc(self.opt_x, get_grad=True,get_hess=doN, penalty_vectors=self.penalty_vector_list)
                
                if self.verbose > 1:
                    print(f"Inner iteration {inner_iter_count}, opt_fx = {self.opt_fx}")
                
                if self._break_condition(inner_iter_count, 'inner'):
                    break
                
                # find step for next iteration
                if doN:
                    Ndir = np.linalg.solve(self.xhess, -self.xgrad)
                    xdir = Ndir / np.linalg.norm(Ndir)
                    last_step_size = last_N_step_size
                    if self.verbose >= 2:
                        print("doing Newton step")
                        #print("xdir dot xgrad is", np.dot(xdir, self.xgrad))
                else:
                    if self.verbose >= 2:
                        print("doing GD step")
                    xdir = -self.xgrad / np.linalg.norm(self.xgrad)
                    last_step_size = last_GD_step_size
                
                opt_step_size, feas_step_size = self._line_search(xdir, self.opt_x, self.opt_fx, self.xgrad, last_step_size)
                last_step_size = self._update_stepsize_add_penalty(opt_step_size, last_step_size, feas_step_size, self.opt_x, xdir, self.opt_fx)
                
                if doN:
                    last_N_step_size = last_step_size
                else:
                    last_GD_step_size = last_step_size
                
                # move on to the next iteration
                self.opt_x += opt_step_size * xdir
                inner_iter_count += 1
            
            # outer iteration check convergence, reduce penalties, and update iter count
            if self._break_condition(outer_iter_count, 'outer'):
                break
            self.opt_params['penalty_ratio'] *= self.opt_params['penalty_reduction']
            outer_iter_count += 1
        
        return self.opt_x, self.opt_fx, self.xgrad, self.xhess
        
