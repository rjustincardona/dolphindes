import pytest
from dolphindes.cvxopt import Alt_Newton_GD, BFGS
import numpy as np 
import scipy.sparse as sp 

np.random.seed(0)

def generate_spd_matrix(n, cond_number=1e3):
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # Random orthogonal matrix
    eigenvalues = np.logspace(0, np.log10(cond_number), n)  # Spread eigenvalues
    A = Q @ np.diag(eigenvalues) @ Q.T  # Construct A with desired conditioning
    return A

@pytest.fixture
def optimization_setup():
    num_dof = 20
    A = generate_spd_matrix(num_dof)
    b = np.random.rand(num_dof)
    opttol = 1e-4
    
    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        objval = x.conj() @ A @ x - b.conj() @ x 
        grad = 2 * A @ x - b if get_grad else []
        hess = 2*A if get_hess else []
        return objval, grad, hess, 0
    
    def feasible_func(x):
        return True
    
    def penalty_func(x):
        pass
    
    opt_config = {'opttol': opttol, 'verbose': 4, 'break_iter_period': 10, 'gradConverge': True}
    
    Ainv = np.linalg.inv(A)
    analytical_solution = 1/2 * Ainv @ b
    
    return {
        'num_dof': num_dof,
        'optfunc': optfunc,
        'feasible_func': feasible_func,
        'penalty_func': penalty_func,
        'opt_config': opt_config,
        'opttol': opttol,
        'analytical_solution': analytical_solution
    }

def test_bfgs_optimum(optimization_setup):
    setup = optimization_setup
    opt = BFGS(setup['optfunc'], setup['feasible_func'], setup['penalty_func'], True, setup['opt_config'])
    opt.run(10*np.array(np.random.random(setup['num_dof'])))
    x, fx = opt.get_last_opt()
    
    assert np.allclose(x, setup['analytical_solution'], atol=setup['opttol'])
    assert np.allclose(fx, setup['optfunc'](setup['analytical_solution'])[0], atol=setup['opttol'])

def test_newton_optimum(optimization_setup):
    setup = optimization_setup
    opt = Alt_Newton_GD(setup['optfunc'], setup['feasible_func'], setup['penalty_func'], True, setup['opt_config'])
    opt.run(10*np.array(np.random.random(setup['num_dof'])))
    x, fx = opt.get_last_opt()
    
    assert np.allclose(x, setup['analytical_solution'], atol=setup['opttol'])
    assert np.allclose(fx, setup['optfunc'](setup['analytical_solution'])[0], atol=setup['opttol'])
