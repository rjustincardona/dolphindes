import pytest
from dolphindes.cvxopt import SharedProjQCQP
import numpy as np 

def test_shared_proj_qcqp_initialization():
    # Initialize with dummy data
    A0 = np.array([[1, 2], [2, 4]])
    s = np.zeros(2)
    c = 0
    A1 = np.array([[1, 2], [2, 4]])
    semidefinite_P = np.array([1, 0])
    projections_diags = np.array([[1, 1], [-1, 1]])

    # Test initialization
    spqcqp = SharedProjQCQP(A0, s, c, A1, semidefinite_P, projections_diags)
    assert spqcqp is not None

    dualval = spqcqp.get_dual(np.array([1, 1, 1]), get_grad=False, get_hess=False)
    print(dualval)
    assert dualval is not None