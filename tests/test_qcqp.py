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


# @pytest.fixture
# def sparse_qcqp_instance():
#     A0 = sp.csc_array(np.array([[1, 2], [2, 4]]))
#     s0 = np.ones(2)
#     c = 0
#     A1 = sp.csc_array(np.array([[1, 2], [2, 4]]))
#     s1 = np.ones(2)/2

#     projections_diags = np.array([[1j, 1j], [-1+0.2j, 2-0.4j]])

#     return SparseSharedProjQCQP(A0, s0, c, A1, s1, projections_diags)

# @pytest.fixture
# def sparse_qcqp_instance():
#     A0 = sp.csc_array(np.array([
#         [1.0, 2.0, 0.0, 0.0],
#         [2.0, 5.0, 1.0, 0.0],
#         [0.0, 1.0, 3.0, 0.5],
#         [0.0, 0.0, 0.5, 4.0]
#     ], dtype=complex))

#     A1 = sp.csc_array(np.array([
#         [4.0, 1.0, 0.0, 0.0],
#         [1.0, 3.0, 1.0, 0.5],
#         [0.0, 1.0, 2.0, 1.0],
#         [0.0, 0.5, 1.0, 2.0]
#     ], dtype=complex))

#     s0 = np.ones(4)
#     s1 = np.ones(4) / 2
#     c = 0.2

#     projections_diags = np.array([
#         [1.0 + 1j, 0.2 + 0.2j, 0.3 + 0.3j, 0.4 + 0.4j],
#         [0.2 - 0.2j, 1.5 + 0.5j, 0.6 + 0.1j, 0.0 + 0.2j],
#         [0.3 - 0.3j, 0.6 - 0.1j, 2.0 + 0j, 0.8 + 0.3j],
#         [0.4 - 0.4j, 0.0 - 0.2j, 0.8 - 0.3j, 1.0 + 0j]
#     ], dtype=complex)

#     return SparseSharedProjQCQP(A0, s0, c, A1, s1, projections_diags)

# def test_projector_addition(sparse_qcqp_instance):
#     assert sparse_qcqp_instance is not None

#     lags = np.array([1.0, 2.0, 3.0, 4.0])
#     combined_projector = sparse_qcqp_instance._add_projectors(lags)
#     assert combined_projector is not None

#     # Manually compute the reference P
#     P_reference = (
#         sparse_qcqp_instance.projections_diags[0] * lags[0] +
#         sparse_qcqp_instance.projections_diags[1] * lags[1] +
#         sparse_qcqp_instance.projections_diags[2] * lags[2] +
#         sparse_qcqp_instance.projections_diags[3] * lags[3]
#     )

#     # Check if the computed projector matches the reference
#     assert np.allclose(combined_projector, P_reference)

# # Test get_dual and compare to dual values
def test_get_dual(data_dir):
    lags = np.load(data_dir / 'ldos_sparse_lags.npy', allow_pickle=True)

    A0 = sp.csc_array(sp.load_npz(data_dir / 'ldos_sparse_A0.npz'))
    A1 = sp.csc_array(sp.load_npz(data_dir / 'ldos_sparse_A1.npz'))
    A2 = sp.csc_array(sp.load_npz(data_dir / 'ldos_sparse_A2.npz'))
    s0 = np.load(data_dir / 'ldos_sparse_s0.npy', allow_pickle=True)
    s1 = np.load(data_dir / 'ldos_sparse_s1.npy', allow_pickle=True)
    projections_diags = np.load(data_dir / 'ldos_some_projections.npy', allow_pickle=True)
    # Interleave the real and imaginary parts of the projections
    real_projs = projections_diags * lags[0::2][:, np.newaxis]
    imag_projs = -1j * projections_diags * lags[1::2][:, np.newaxis]
    
    # Create interleaved array
    interleaved_projs = np.empty((real_projs.shape[0] + imag_projs.shape[0], projections_diags.shape[1]), dtype=complex)
    interleaved_projs[0::2] = real_projs
    interleaved_projs[1::2] = imag_projs
    
    projections_diags = interleaved_projs

    c = 0.0 
    c1 = 0.0 
    lags = np.ones(len(projections_diags)) 

    assert projections_diags.shape[0] == len(lags), "Number of projections does not match number of lags."

    sparse_ldos_qcqp = SparseSharedProjQCQP(A0, s0, c, A1, A2, s1, c1, projections_diags)
    combined_projector = sp.diags(sparse_ldos_qcqp._add_projectors(lags))

    ref_totalA = sp.load_npz(data_dir / 'ldos_sparse_totalA.npz')
    calc_totalA = sparse_ldos_qcqp._get_total_A(combined_projector)
    assert sp.issparse(calc_totalA), "Total A should be a sparse matrix."
    assert sp.csr_array(calc_totalA).shape == ref_totalA.shape, "Shape of calculated total A does not match reference."
    assert np.allclose(calc_totalA.toarray(), ref_totalA.toarray()), "Calculated total A does not match reference total A."

    dual, grad, hess = sparse_ldos_qcqp.get_dual(lags, get_grad=True)
    # assert dual is not None
    # assert np.isclose(dual, 12, atol=0.5)

    print(dual)
    print(grad)
    # pass 
    # # Load the reference dual values
    # reference_dual = np.load('refernce_arrays/qcqp_example/ldos_sparse_dual.npy')

    # # Check if the computed dual matches the reference
    # assert np.allclose(dual, reference_dual, atol=1e-6), "Dual values do not match reference values."
    
