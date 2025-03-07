import pytest
from dolphindes.cvxopt import Newton, BFGS
import numpy as np 
import scipy.sparse as sp 

# @pytest.fixture
# def optfunc_opt_boundary():
#     def func(x):
#         return np.sum(x**2) + 1
#     return func

def test_bfgs_optimum():
    def optfunc(x, get_grad=True, get_hess=False):
        objval = x.conj() @ x + 1
        grad = 2 * x
        hess = 2 * np.eye(len(x))
        return objval, grad, hess
    
    opttol = 1e-6
    opt = BFGS(optfunc, lambda x: True, lambda x: np.zeros_like(x), {'opttol': opttol, 'verbose': 3})
    opt.run(10*np.array(np.random.random(2)))
    x, fx = opt.get_last_opt()
    assert np.allclose(x, 0.0, atol=opttol)
    assert np.allclose(fx, 1.0, atol=opttol)

def test_newton_optimum():
    pass 

def test_newton_tolerance():
    pass 

def test_newton_max_and_min_iter():
    pass 
