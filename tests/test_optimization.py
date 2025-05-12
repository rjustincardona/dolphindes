import pytest
from dolphindes.cvxopt import Newton, BFGS
import numpy as np 
import scipy.sparse as sp 

np.random.seed(0)

def generate_spd_matrix(n, cond_number=1e3):
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # Random orthogonal matrix
    eigenvalues = np.logspace(0, np.log10(cond_number), n)  # Spread eigenvalues
    A = Q @ np.diag(eigenvalues) @ Q.T  # Construct A with desired conditioning
    return A

def test_bfgs_optimum():
    num_dof = 20
    A = generate_spd_matrix(num_dof)
    b = np.random.rand(num_dof)
    
    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        objval = x.conj() @ A @ x - b.conj() @ x 
        grad = 2 * A @ x - b if get_grad else []
        hess = 2*A if get_hess else []
        return objval, grad, hess, 0
    
    opttol = 1e-4
    
    def feasible_func(x):
        return True 
    
    def penalty_func(x):
        pass 
    
    opt = BFGS(optfunc, feasible_func, penalty_func, True, {'opttol': opttol, 'verbose': 4, 'break_iter_period': 10, 'gradConverge': True})
    opt.run(10*np.array(np.random.random(num_dof)))
    x, fx = opt.get_last_opt()
    # print(x, fx)
    Ainv = np.linalg.inv(A)
    xstar = 1/2 * Ainv @ b

    assert np.allclose(x, xstar, atol=opttol)
    assert np.allclose(fx, optfunc(xstar)[0], atol=opttol)

def test_newton_optimum():
    pass 
