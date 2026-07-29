"""
Matrix-free (CG-based, log-barrier) alternative to Cholesky-based QCQP solving.

Opt in via `Photonics_FDFD(..., matrix_free=True)`; `setup_QCQP` then builds a
`MatrixFreeSharedProjQCQP` instead of `SparseSharedProjQCQP`/
`DenseSharedProjQCQP`, exposing the same public solving interface
(`solve_current_dual_problem`, `run_gcd`, `is_dual_feasible`, `get_dual`).
Problem assembly (A0/A1/A2/s0/s1) is unaffected -- only the solver changes.
"""

from .gcd import MatrixFreeGCDParams
from .qcqp import MatrixFreeBFGSParams, MatrixFreeDualResult, MatrixFreeSharedProjQCQP

__all__ = [
    "MatrixFreeSharedProjQCQP",
    "MatrixFreeGCDParams",
    "MatrixFreeBFGSParams",
    "MatrixFreeDualResult",
]
