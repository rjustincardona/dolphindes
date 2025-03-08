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

    def optfunc(x, get_grad=False, get_hess=False):
        objval = x.conj() @ A @ x - b.conj() @ x
        grad = 2 * A @ x - b if get_grad else []
        hess = [] #2 * np.eye(len(x))
        return objval, grad, hess
    
    opttol = 1e-4
    
    opt = BFGS(optfunc, lambda x: True, lambda x: np.zeros_like(x), True, {'opttol': opttol, 'verbose': 0, 'break_iter_period': 100})
    opt.run(10*np.array(np.random.random(num_dof)))
    x, fx = opt.get_last_opt()

    Ainv = np.linalg.inv(A)
    xstar = 1/2 * Ainv @ b

    assert np.allclose(x, xstar, atol=opttol)
    assert np.allclose(fx, optfunc(xstar)[0], atol=opttol)

def test_newton_optimum():
    pass 

def test_newton_tolerance():
    pass 

def test_newton_max_and_min_iter():
    pass 
