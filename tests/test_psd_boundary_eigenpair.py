"""Tests for the PSD-boundary eigenpair used to build penalty vectors.

``_get_PSD_penalty`` returns the eigenvalue of A(lags) closest to zero and its
eigenvector, found by shift-invert ``eigsh`` at sigma=0. Shift-invert has to apply
A^{-1}, and the sparse formulation supplies its existing CHOLMOD factor for that
through ``_shift_invert_OPinv`` rather than letting ``eigsh`` build its own
factorization.
"""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP


def _problem_data(rng, n):
    """Return matrices for a QCQP that is dual feasible at zero multipliers."""
    dense = rng.random((n, n)) + 1j * rng.random((n, n))
    dense[rng.random((n, n)) > 0.4] = 0.0
    A1 = dense + dense.conj().T
    # A0 positive definite but with a small eigenvalue, so the smallest-magnitude
    # eigenvalue is well separated from the rest and unambiguous to identify.
    Q, _ = la.qr(rng.random((n, n)) + 1j * rng.random((n, n)))
    spectrum = np.concatenate([[1e-3], 1.0 + rng.random(n - 1)])
    A0 = Q @ np.diag(spectrum) @ Q.conj().T
    A0 = (A0 + A0.conj().T) / 2
    s0 = rng.random(n) + 1j * rng.random(n)
    s1 = rng.random(n) + 1j * rng.random(n)
    Pdiags = [rng.random(n) + 1j * rng.random(n) for _ in range(3)]
    return A0, A1, s0, s1, Pdiags


@pytest.fixture(params=["sparse", "dense"])
def qcqp(request):
    """Return a small QCQP in the sparse or dense formulation."""
    rng = np.random.default_rng(17)
    n = 10
    A0, A1, s0, s1, Pdiags = _problem_data(rng, n)
    if request.param == "sparse":
        Plist = [sp.diags_array(p, format="csc") for p in Pdiags]
        return SparseSharedProjQCQP(
            sp.csc_array(A0),
            s0,
            0.0,
            sp.csc_array(A1),
            sp.eye_array(n, dtype=complex, format="csc"),
            s1,
            Plist,
            sp.diags_array(np.ones(n, dtype=complex), format="csc"),
            verbose=0,
        )
    Plist = [sp.diags_array(p, format="csc") for p in Pdiags]
    return DenseSharedProjQCQP(
        A0,
        s0,
        0.0,
        A1,
        s1,
        Plist,
        sp.diags_array(np.ones(n, dtype=complex), format="csc"),
        verbose=0,
    )


def _dense_A(qcqp, lags):
    """Return A(lags) as a dense array."""
    A = qcqp._get_total_A(lags)
    return A.toarray() if sp.issparse(A) else np.asarray(A)


def test_returns_the_eigenvalue_closest_to_zero(qcqp):
    """The returned pair is the smallest-magnitude eigenpair of A(lags).

    Checked against a full dense eigendecomposition. This is what would fail if
    the operator handed to shift-invert did not actually apply A^{-1}.
    """
    lags = np.zeros(qcqp.get_number_constraints())
    assert qcqp.is_dual_feasible(lags)

    v, lam = qcqp._get_PSD_penalty(lags)
    A = _dense_A(qcqp, lags)
    reference = la.eigvalsh(A)
    closest = reference[np.argmin(np.abs(reference))]

    assert np.isclose(lam, closest, rtol=1e-8, atol=1e-12), (
        f"got {lam}, dense spectrum has {closest} closest to zero"
    )
    # A feasible point is positive definite, so the eigenvalue must be positive.
    assert lam > 0


def test_returned_vector_is_an_eigenvector(qcqp):
    """The vector satisfies A v = lam v, and is normalized."""
    lags = np.zeros(qcqp.get_number_constraints())
    v, lam = qcqp._get_PSD_penalty(lags)
    A = _dense_A(qcqp, lags)

    assert np.isclose(np.linalg.norm(v), 1.0, rtol=1e-8)
    residual = np.linalg.norm(A @ v - lam * v)
    assert residual < 1e-8 * max(np.abs(lam), 1.0) + 1e-12


def test_agrees_with_letting_eigsh_factorize(qcqp):
    """Supplying A^{-1} gives the same eigenpair as letting ``eigsh`` build it.

    The two differ only in how A^{-1} is applied, so the eigenvalue must match
    tightly and the eigenvectors must be parallel: they are defined only up to
    phase, so compare the overlap rather than the components.
    """
    lags = np.zeros(qcqp.get_number_constraints())
    A, _ = qcqp._get_factorization(lags)

    v_supplied, lam_supplied = qcqp._get_PSD_penalty(lags)
    lam_own, v_own = spla.eigsh(
        A, k=1, sigma=0.0, which="LM", return_eigenvectors=True
    )

    assert np.isclose(lam_supplied, lam_own[0], rtol=1e-8)
    overlap = np.abs(np.vdot(v_own[:, 0], v_supplied)) / (
        np.linalg.norm(v_own[:, 0]) * np.linalg.norm(v_supplied)
    )
    assert np.isclose(overlap, 1.0, atol=1e-8)


def test_only_the_sparse_formulation_supplies_an_inverse_operator(qcqp):
    """The sparse path supplies CHOLMOD; the dense path deliberately declines.

    Not an accident to be tidied up later: on the dense formulation ``lu_solve``
    applies A^{-1} about three times faster per application than ``cho_solve``, so
    letting ``eigsh`` build its own LU beats reusing the Cholesky factor even
    though the factorization is already paid for.
    """
    lags = np.zeros(qcqp.get_number_constraints())
    A, _ = qcqp._get_factorization(lags)
    operator = qcqp._shift_invert_OPinv(A)

    if isinstance(qcqp, SparseSharedProjQCQP):
        assert isinstance(operator, spla.LinearOperator)
        # It must apply A^-1, not A.
        rng = np.random.default_rng(1)
        b = rng.random(A.shape[0]) + 1j * rng.random(A.shape[0])
        dense = _dense_A(qcqp, lags)
        assert np.allclose(dense @ (operator @ b), b, atol=1e-8)
    else:
        assert operator is None
