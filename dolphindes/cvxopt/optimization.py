"""
Optimizers

"""

__all__ = ["BFGS", "Newton"]

import numpy as np 

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
        opt_params: Dictionary containing optimization parameters. These parameters 
            override the default parameters defined in OPT_PARAMS_DEFAULTS.
    
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

    OPT_PARAMS_DEFAULTS = {'opttol': 1e-6, 'gradConverge': False, 'min_inner_iter': 5, 'max_restart': np.inf, 'penalty_ratio': 1e-2, 'penalty_reduction': 0.1, 'break_iter_period': 50, 'verbose': 0, 'penalty_vector_list': []}

    def __init__(self, optfunc, feasible_func, penalty_vector_func, is_convex, opt_params):
        self.optfunc = optfunc
        self.feasible_func = feasible_func
        self.penalty_vector_func = penalty_vector_func
        self.penalty_vector_list = [] 
        self.is_convex = is_convex
        self.opt_params = {**self.OPT_PARAMS_DEFAULTS, **opt_params}

        if self.opt_params['verbose'] > 0:
            print('Optimizer initialized with parameters:')
            for k, v in self.opt_params.items():
                print(f'{k}: {v}')

        self.opt_x = None 
        self.opt_fx = None
        self.verbose = self.opt_params['verbose']

    def get_last_opt(self):
        """Get the last optimized x and f(x) values."""
        return self.opt_x, self.opt_fx
    
    def _line_search(self, dir : np.ndarray, x0 : np.ndarray, fx0 : np.ndarray, grad : np.ndarray, init_step_size : float, add_penalty : bool = False) -> float:
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

        # Next, find optimal alpha 
        c_A = 1e-4
        opt_val = np.inf
        grad_direction = dir @ grad
        while True:
            tmp_value, _, _ = self.optfunc(x0 + alpha*dir, get_grad=False, get_hess=False)
            if tmp_value < opt_val: #the dual is still decreasing as we backtrack, continue
                opt_val = tmp_value; alpha_opt=alpha
            else:
                break
            if not self.is_convex and tmp_value <= fx0 + c_A * alpha * grad_direction: #Armijo backtracking condition if problem is not convex
                alpha_opt = alpha
                break
            
            alpha *= c_reduct
        
        if add_penalty:
            pass 

        if self.verbose >= 2: print(f"Line search found optimal step size: {alpha_opt}")

        return alpha_opt
    
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
        - _line_search() is implemented to find optimal step size
        - _update_hess_inv() is implemented to update the inverse Hessian approximation as part of BFGS
    """
    def __init__(self, *params):
        super().__init__(*params)
    
    def _break_condition(self, iter_num, iter_type):
        if iter_type == 'inner':
            #TODO(alessio): implement gradconverge or objval termination

            if iter_num % self.opt_params['break_iter_period'] == 0 and iter_num > self.opt_params['min_inner_iter']:
                if np.abs(self.prev_fx - self.opt_fx) < np.abs(self.opt_fx)*self.opt_params['opttol']:
                    return True
                self.prev_fx = self.opt_fx

        elif iter_type == 'outer':
            raise NotImplementedError("Outer break condition not implemented yet")
        
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
        if abs(gamma_dot_delta) < 1e-10:
            if self.verbose >= 3:
                print("Skipping Hinv update due to small gamma_dot_delta")
            return Hinv
            
        # Standard BFGS update formula
        rho = 1.0 / gamma_dot_delta
        I = np.eye(Hinv.shape[0])
        term1 = I - rho * np.outer(delta, gamma)
        term2 = I - rho * np.outer(gamma, delta)
        term3 = rho * np.outer(delta, delta)
        
        new_Hinv = term1 @ Hinv @ term2 + term3
            
        return new_Hinv
    
    def run(self, x0, tol=None):
        self.opt_x = x0
        self.ndof = self.opt_x.size
        self.xgrad = np.zeros(self.ndof)
        self.prev_fx = np.inf

        if tol is None: tol = self.opt_params['opttol']

        outer_iter_count = 0
        inner_iter_count = 0

        if self.verbose > 0:
                print(f"Starting optimization with x0 = {self.opt_x}")

        while True: # outer loop - penalty reduction
            self.penalty_vector_list = [] # reset penalty vectors
            self.opt_fx, self.xgrad, _ = self.optfunc(self.opt_x, get_grad=True, get_hess=False)

            last_step_size = 1.0 
            Hinv = np.eye(self.ndof)

            inner_iter_count = 0 

            if self.verbose > 0:
                print(f"Outer iteration {outer_iter_count}, penalty_ratio = {self.opt_params['penalty_ratio']}, opt_fx = {self.opt_fx}")

            while True:
                inner_iter_count += 1

                if self.opt_params['verbose'] > 1:
                    print(f"Inner iteration {inner_iter_count}, opt_fx = {self.opt_fx}")

                BFGS_dir = - Hinv @ self.xgrad
                nBFGS_dir = BFGS_dir / np.linalg.norm(BFGS_dir)

                opt_step_size = self._line_search(nBFGS_dir, self.opt_x, self.opt_fx, self.xgrad, last_step_size) # Perform line search for step size 
                
                delta = opt_step_size * nBFGS_dir
                old_grad = self.xgrad.copy()  # Save old gradient before update
                self.opt_x += delta

                new_opt_fx, new_grad, _ = self.optfunc(self.opt_x, get_grad=True, get_hess=False)

                Hinv = self._update_Hinv(Hinv, new_grad, old_grad, delta) # Update Hessian inverse 
                
                self.opt_fx = new_opt_fx
                self.xgrad = new_grad

                if self._break_condition(inner_iter_count, 'inner'): break 

            # if self._break_condition('outer'): break 
            break

            outer_iter_count += 1
            self.opt_params['penalty_ratio'] *= self.opt_params['penalty_reduction']
            
        return self.opt_x, self.xgrad, self.opt_fx

class Newton(_Optimizer):
    def __init__(self, *params):
        super().__init__(*params)
        self.hess = np.zeros((self.ndof, self.ndof))
        raise NotImplementedError("Newton not implemented yet")



