"""
Optimizers

"""

__all__ = ["BFGS", "Alt_Newton_GD"]

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import AAA
from numpy.linalg import norm
import matplotlib.pyplot as plt

def bisect(f, a, b, max_iter=50, tol=1e-4):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        if fb < 0:
            # print(f"trying again at a={b} and b={2*b}, {fb}")
            # input()
            return bisect(f, b, 2*b, max_iter, tol)
        raise ValueError        
    for it in range(max_iter):
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
    print(it)
    return (a + b) / 2

def root_aaa(f, x0, max_iter=10, max_restart=10, r = 1e-2, tol=1e-4, verbose=False):
    z = x0
    for _ in range(max_restart):
        xs = [z]
        fs = [f(z)]

        xs.append(x0 - r * np.sign(fs[-1]))
        fs.append(f(xs[-1]))
        for iter in range(max_iter):
            a = AAA(xs, fs)
            try:
                z = np.max(np.real(a.roots()))
            except ValueError:
                # print(xs, fs)
                raise ValueError
            err = np.real(f(z))
            # if verbose:
            #     print(f"\troot_aaa: z = {z}, err = {err}")
            if abs(err) < tol:
                return z#, np.max(np.real(a.poles()))
            xs.append(z)
            fs.append(err)
        z = xs[np.argmin(np.abs(np.array(fs)))]
    # print("trying further away", xs, fs)
    # input()
    # if verbose:
    #     x_plot = np.linspace(0, x0+10, 10000)
    #     y_plot = [f(a) for a in x_plot]
    #     plt.plot(x_plot, y_plot)
    #     plt.show()
    return root_aaa(f, np.real(np.max(a.poles())) + 10, max_iter, max_restart, r, tol, verbose=verbose)

def pole_aaa(f, start, end, n=10):
    xs = np.linspace(start, end, n)
    fs = [f(x) for x in xs]
    a = AAA(xs, fs)
    return np.max(np.real(a.poles())).real

# def pole_aaa(f, x0, max_iter=10, max_restart=10, r = 1e-2, tol=1e-4, verbose=False):
#         z = x0
#         xs = np.linspace(z-r, z+r, 5)
#         fs = [f(x) for x in xs]
#         a = AAA(xs, fs)
#         return np.max(np.real(a.poles())).real
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

    def __init__(self, optfunc, aU, A1, A2, Pdiags, _get_total_A, _get_total_S, _add_projectors, s1, _get_xstar, feasible_func, penalty_vector_func, is_convex, opt_params, g=None, a=None):
        self.optfunc = optfunc

        self.aU = aU
        self.A1 = A1
        self.A2 = A2
        self.Pdiags = Pdiags
        self._get_total_A = _get_total_A
        self._get_total_S = _get_total_S
        self._add_projectors = _add_projectors
        self.s1 = s1
        self._get_xstar = _get_xstar
        self.g=g
        self.a=a

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

    def C(self, z, L):
            x = L.copy()
            x[1] = z
            return self.optfunc(x, get_grad=True,get_hess=False, penalty_vectors=self.penalty_vector_list)[1][1].real
    def D(self, z, L):
            x = L.copy()
            x[1] = z
            xstar, dualval = self._get_xstar(x)
            A2_xstar = self.A2 @ xstar      # Shape: (N_p,)
            x_conj_A1 = xstar.conj() @ self.A1  # Shape: (N_p,) where N_p = self.A1.shape[1]
            term1 = -np.real( (x_conj_A1 * A2_xstar) @ self.Pdiags )

            term2 = 2 * np.real( (A2_xstar.conj() * np.ones(self.s1.shape)) @ self.Pdiags.conj() )
            grad = term1 + term2
            return grad[1].real
    def root(self, x0):
        x = x0.copy()
        x[1] = 0
        pole = -spla.eigs(self._get_total_A(x), M=self.aU, k=1, return_eigenvectors=False)[0].real
        if pole < 0:
            pole = -spla.eigs(self._get_total_A(x), M=self.aU, sigma=pole, k=1, return_eigenvectors=False)[0].real

        rad = 1e-5
        try:
            return bisect(lambda a: self.C(a, x), pole, 2 * pole)
        except ValueError:
            while True:
                try:
                    return root_aaa(lambda a: self.C(a, x), pole-rad, r=rad/10)[0]
                except ValueError:
                    rad /= 10

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
        while not self.feasible_func(x0 + alpha * dir):
            alpha *= c_reduct
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

        return alpha_opt, alpha_feas
    
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
                
                if gradConverge  and np.abs(function_value - fminus_xxgrad)< opttol * np.abs(fminus_xxgrad) and np.abs(remaining_descent)< opttol * np.abs(fminus_xxgrad) and norm(self.xgrad) < opttol * np.abs(function_value): 
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
    
    def _add_penalty(self, opt_step_size, last_step_size, feas_step_size, x0, dir, opt_fx0):
        if np.isclose(opt_step_size, last_step_size, atol=0.0):
            # if no backtracking happened, can start with a more aggressive stepsize
            return opt_step_size * 2, False
        else:
            if feas_step_size < last_step_size and np.isclose(opt_step_size, feas_step_size, atol=0.0):
                # all the backtracking due to feasibility reasons, add penalty
                if self.verbose >= 2: print("Adding penalty due to feasibility wall.")
                penalty_vector, _ = self.penalty_vector_func(x0 + opt_step_size * dir)
                penalty_value = self.optfunc(x0, get_grad=False, get_hess=False, penalty_vectors=[penalty_vector])[0] 
                epsS = np.sqrt(self.opt_params['penalty_ratio']*np.abs(opt_fx0 / penalty_value))
                self.penalty_vector_list.append(epsS * penalty_vector)

                return opt_step_size, True
            return opt_step_size, False

    def run(self, x_init, g_init=None, alpha_init = 1e-5, alpha_min=1e-7, grad_tol=1e-1, stable_n=5, stable_tol=1e-3):
        # Initialize in PD region
        if g_init is None:
            g_init = self.root(x_init)
        x_init[1] = max(0, g_init)
        
        # Initialize history
        alpha = [alpha_init]
        g = [g_init]
        x = [x_init]
        dual = []
        grad = []
        hess = []
        ndir = []
        dual_init, grad_init, _, _ = self.optfunc(x_init, get_grad=True, get_hess=False)
        grad_init[1] = 0
        hess_init = np.eye(x_init.size)
        ndir_init = grad_init / norm(grad_init)
        dual.append(dual_init)
        grad.append(grad_init)
        hess.append(hess_init)
        ndir.append(ndir_init)

        iters = 0
        reverting = False
        force_power_iter = 0
        while True:
            iters += 1
            alpha_new = alpha[-1]
            
            # Revert if needed
            if reverting:
                iters -= 1
                g_true = self.root(x[-1])
                if np.abs(g_true - g[-1]) < 1e-2:
                    force_power_iter += 1
                    reverting = False
                else:
                    if iters > 1:
                        iters -= 1
                        alpha.pop()
                        g.pop()
                        x.pop()
                        dual.pop()
                        grad.pop()
                        hess.pop()
                        ndir.pop()
                        continue
                    return self.run(x_init, g_init=g_true, alpha_init=alpha_min, alpha_min=alpha_min, grad_tol=grad_tol)
            else:
                force_power_iter = max(0, force_power_iter - 1)

            # Determine step
            iters_back = 0
            while True:
                iters_back += 1

                # Propose new x
                x_new = x[-1] + alpha_new * ndir[-1]
                try:
                    if force_power_iter > 0:
                        raise ValueError("forcing power iteration.")
                    g_new = root_aaa(lambda z: self.C(z, x_new), g[-1])
                except ValueError:
                    g_new = self.root(x_new)
                x_new[1] = max(0, g_new)

                # Get state at new x
                dual_new, grad_new, _, _ = self.optfunc(x_new, get_grad=True, get_hess=False)
                grad_new[1] = 0
                hess_new = self._update_Hinv(hess[-1], grad_new, grad[-1], ndir[-1], reset=False)
                hess_small = np.vstack((hess_new[0], hess_new[2:]))
                hess_small = np.vstack((hess_new.T[0], hess_new.T[2:])).T
                grad_small = np.concatenate(([grad_new[0]], grad_new[2:]))
                ndir_new = -hess_small @ grad_small
                ndir_new = ndir_new / norm(ndir_new)
                print(f"\titer {iters}-{iters_back}-> dual: {dual_new:.4f}, grad: {norm(grad_new):.4f}, alpha: {alpha_new}, power {force_power_iter}")

                # Validate new state
                # e = np.min(np.linalg.eigvals(self._get_total_A(x_new).todense()).real)
                # if e < 0:
                #     print(f"\t\t Indefinite with lowest eigenvalue {e:.4f}")
                if alpha_new < alpha_min:
                    alpha_new = alpha_init
                elif dual_new > dual[-1]:
                    alpha_new *= 1e-1
                    continue
                elif norm(grad_new) > norm(grad[-1]):
                    if not force_power_iter:
                        reverting = True
                        break
                

                # Choose new parameters
                if iters_back == 1:
                    alpha_new *= 1e1
                alpha.append(alpha_new)
                g.append(g_new)
                x.append(x_new)
                dual.append(dual_new)
                grad.append(grad_new)
                hess.append(hess_new)
                ndir.append(ndir_new)
                break
        
            # Check convergence
            if norm(grad[-1]) < grad_tol or (iters > stable_n and np.abs(dual[-1] - np.mean(dual[-stable_n:-1])) < stable_tol):
                return x[-1], dual[-1], grad[-1], None, g[-1], alpha[-1]


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
                    print(f"opt_fx: {self.opt_fx}, fminus_xxgrad: {fminus_xxgrad}, grad norm: {norm(self.xgrad)}")
                    
                if gradConverge  and np.abs(function_value - fminus_xxgrad)< opttol * np.abs(fminus_xxgrad) and np.abs(remaining_descent)< opttol * np.abs(fminus_xxgrad) and norm(self.xgrad) < opttol * np.abs(function_value): 
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
                    xdir = Ndir / norm(Ndir)
                    last_step_size = last_N_step_size
                    if self.verbose >= 2:
                        print("doing Newton step")
                        #print("xdir dot xgrad is", np.dot(xdir, self.xgrad))
                else:
                    if self.verbose >= 2:
                        print("doing GD step")
                    xdir = -self.xgrad / norm(self.xgrad)
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
        
