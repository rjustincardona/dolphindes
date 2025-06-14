import pytest
from dolphindes.cvxopt import SparseSharedProjQCQP
import numpy as np 
import scipy.sparse as sp 
import os 
from pathlib import Path

@pytest.fixture
def data_dir():
    """Return the path to the reference data directory."""
    return Path(os.path.dirname(__file__)) / "reference_arrays" / "qcqp_example"

@pytest.fixture(params=['global', 'local'])
def sparse_qcqp_data(data_dir, request):
    """Fixture to load data and initialize SparseSharedProjQCQP."""
    added_str = request.param
    print(f"\nLoading data for: {added_str} constraints")
    data_path = data_dir / added_str
    lags = np.load(data_path / 'ldos_sparse_lags.npy', allow_pickle=True)
    lags_optimal = np.load(data_path / 'ldos_sparse_lags_optimal.npy', allow_pickle=True)
    A0 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A0.npz'))
    A1 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A1.npz'))
    A2 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A2.npz'))
    s0 = np.load(data_path / 'ldos_sparse_s0.npy', allow_pickle=True)
    s1 = np.load(data_path / 'ldos_sparse_s1.npy', allow_pickle=True)
    projections_diags = np.load(data_path / 'ldos_some_projections.npy', allow_pickle=True)
    # Interleave projections_diags and projections_diags * -1j
    projections_diags = np.asarray(projections_diags)
    interleaved = np.empty((2 * projections_diags.shape[0], projections_diags.shape[1]), dtype=complex)
    interleaved[0::2] = projections_diags
    interleaved[1::2] = projections_diags * -1j
    projections_diags = interleaved
    Pdiags = projections_diags.T

    c = np.load(data_path / 'ldos_dualconst.npy', allow_pickle=True)
    
    sparse_ldos_qcqp_instance = SparseSharedProjQCQP(A0, s0, c, A1, A2, s1, Pdiags, verbose=0)
    
    return {
        "qcqp": sparse_ldos_qcqp_instance,
        "lags": lags,
        "lags_optimal": lags_optimal,
        "A0": A0, "A1": A1, "A2": A2,
        "s0": s0, "s1": s1,
        "Pdiags": Pdiags,
        "c": c,
        "data_path": data_path
    }

def test_sparse_qcqp(sparse_qcqp_data):
    """ 
    Test QCQP optimization with BFGS, and compare with optimized lags and results. This doesn't combine Lags into projectors, so gradients and penalties should be the same 
    as known cases. 
    """
    sparse_ldos_qcqp = sparse_qcqp_data["qcqp"]
    lags = sparse_qcqp_data["lags"]
    lags_optimal = sparse_qcqp_data["lags_optimal"]
    data = sparse_qcqp_data["data_path"]
    c1 = 0.0 

    combined_projector = sparse_ldos_qcqp._add_projectors(lags)

    print("Testing totalA = known total A")
    ref_totalA = sp.load_npz(data / 'ldos_sparse_totalA.npz')
    calc_totalA = sparse_ldos_qcqp._get_total_A(lags)
    assert sp.issparse(calc_totalA), "Total A should be a sparse matrix."
    assert sp.csr_array(calc_totalA).shape == ref_totalA.shape, "Shape of calculated total A does not match reference."
    assert np.allclose(calc_totalA.toarray(), ref_totalA.toarray()), "Calculated total A does not match reference total A."

    print("Testing totalS = known total S")
    ref_totals = np.load(data / 'ldos_sparse_total_s.npy', allow_pickle=True)
    calc_totals = sparse_ldos_qcqp._get_total_S(combined_projector)
    assert calc_totals.shape == ref_totals.shape, "Shape of calculated total S does not match reference."
    assert np.allclose(calc_totals, ref_totals), "Calculated total S does not match reference total S."

    print("Testing dualval = known dualval and grad = known grad, as well as penalty vector dualval and grad")
    init_lags = sparse_ldos_qcqp.find_feasible_lags()
    dual, grad, hess, aux = sparse_ldos_qcqp.get_dual(init_lags, get_grad=True)
    assert dual is not None
    print(f'some feasible dual : {dual}')
    feasible = sparse_ldos_qcqp.is_dual_feasible(init_lags)
    assert feasible, "Dual is not feasible."

    dual_opt, grad_opt, hess_opt, aux_opt = sparse_ldos_qcqp.get_dual(lags_optimal, get_grad=True, get_hess=False)
    assert dual is not None
    print(f'optimal dual : {dual_opt}')
    print(np.load(data / 'ldos_dualval.npy'))
    assert np.allclose(dual_opt, np.load(data / 'ldos_dualval.npy'), atol=1e-6), "Dual values do not match."
    assert np.allclose(grad_opt, np.load(data / 'ldos_optgrad.npy'), atol=1e-6), "Gradients do not match."

    print("Testing dualvalue of penalty vector")
    penalty, value = sparse_ldos_qcqp._get_PSD_penalty(lags_optimal)
    dual_opt_penalty, grad_opt_penalty, hess_opt_penalty, aux_opt_penalty = sparse_ldos_qcqp.get_dual(lags_optimal, get_grad=True, penalty_vectors=[penalty])
    assert np.allclose(aux_opt_penalty.dualval_penalty, np.load(data / 'ldos_dualval_penalty.npy'), atol=1e-6), "Penalty dual values do not match."
    assert np.allclose(aux_opt_penalty.grad_penalty, np.load(data / 'ldos_grad_penalty.npy'), atol=1e-6), "Penalty gradients do not match."

    print("Testing solving the dual problem with BFGS and Newton")
    current_dual, dual_lambda, current_grad, current_hess, xstar = sparse_ldos_qcqp.solve_current_dual_problem('bfgs', init_lags = init_lags)

    print(f'dual lambda: {dual_lambda}')
    print(f'bound BFGS: {current_dual}')

    assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), "Dual lambda does not match optimal lags."
    assert np.allclose(current_dual, dual_opt, atol=1e-2), "Dual values does not match optimal value."

    current_dual, dual_lambda, current_grad, current_hess, xstar = sparse_ldos_qcqp.solve_current_dual_problem('newton', init_lags = init_lags)
    print(f'dual lambda newton: {dual_lambda}')
    print(f'bound Newton: {current_dual}')

    assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), "Newton dual lambda does not match optimal lags."
    assert np.allclose(current_dual, dual_opt, atol=1e-4), "Newton dual values does not match optimal value."
    dual_opt, grad_opt, hess_opt, aux_opt = sparse_ldos_qcqp.get_dual(dual_lambda, get_grad=True, get_hess=True)
    assert np.allclose(hess_opt, np.load(data / 'ldos_opthess.npy'), rtol=1e-2), "Hessian does not match optimal Hessian." # Hessian values can be large, so allow some tolerance.

    print("Testing iterative splitting step")
    results = [] 
    result_counter = 0
    for result in sparse_ldos_qcqp.iterative_splitting_step():
        results.append((sparse_ldos_qcqp.Pdiags.shape[1], result[0]))
        result_counter += 1
        if result_counter > 0:
            assert results[result_counter-1][1] >= result[0], "Iterative splitting step must decrease dualval."
        if sparse_ldos_qcqp.Pdiags.shape[1] > 200: # limit to avoid excessive iterations. Run entire test for most rigorous test. 
            break
    print(results)

    print("Testing the merging of constraints")
    from dolphindes.cvxopt import merge_lead_constraints
    dual_opt, dual_grad, dual_hess, _ = sparse_ldos_qcqp.get_dual(sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True)
    
    merge_lead_constraints(sparse_ldos_qcqp, 5)
    merged_dual, merged_grad, merged_hess, _ = sparse_ldos_qcqp.get_dual(sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True)
    assert np.allclose(dual_opt, merged_dual, atol=1e-2) , "dual value changed after constraint merge."
    assert np.allclose(dual_grad[-5:], merged_grad[-5:], atol=1e-2), "dual grad changed after constraint merge."
    assert np.allclose(dual_hess[-5:,-5:], merged_hess[-5:,-5:], rtol=1e-2), "dual Hess changed after constraint merge."
    