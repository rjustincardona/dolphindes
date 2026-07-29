"""
Design problem data and its matrix-free operator kernels.

A DesignProblem carries the quadratic-program data produced by dolphindes's
existing problem-assembly code (e.g. Photonics_FDFD.setup_QCQP) -- or by any
other solver entirely -- and knows how to apply the Lagrangian operators
without assembling them:

    maximize_x   -x^† A0 x + 2 Re(x^† s0) + c0
    subject to   Re( -x^† A1 P_j A2 x + 2 x^† A2^† P_j^† s1 ) = 0

with DIAGONAL projectors P_j. For a combined diagonal d = Pdiags @ lags,

    A(lags) x = A0 x + 1/2 * ( A1 (d ⊙ (A2 x)) + A2^† (conj(d) ⊙ (A1^† x)) )

which costs five matvecs regardless of the number of constraints.

A0, A1, A2 may be ANY linear operator, not just scipy.sparse matrices --
numpy arrays, scipy.sparse matrices/arrays, scipy.sparse.linalg.LinearOperator
instances, or your own class -- as long as it supports `@` (a matvec) and has
a `.shape` attribute. For A1/A2, the Hermitian adjoint is derived
automatically: `op.conj().T` if supported (numpy arrays, scipy.sparse
matrices/arrays all support this), else `op.H` (the convention
scipy.sparse.linalg.LinearOperator uses instead, since LinearOperator has no
`.conj()`). If your own operator supports neither, give it a `.conj().T` or
`.H` that returns its adjoint (or just wrap it in a LinearOperator with an
explicit `rmatvec`, which does this for you). A0 is optional (default None,
meaning the exact zero matrix -- e.g. the LDOS problem) and its matvec is
skipped entirely when omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


def _adjoint(op: Any) -> Any:
    """Hermitian adjoint of any linear operator.

    Tries `op.conj().T` first (numpy arrays, scipy.sparse matrices/arrays);
    falls back to `op.H` (scipy.sparse.linalg.LinearOperator's convention,
    which has no `.conj()` of its own).
    """
    try:
        return op.conj().T
    except AttributeError:
        return op.H


@dataclass
class DesignProblem:
    """Quadratic-program data: A0/A1/A2 any linear operator, vectors complex.

    Adjoints (A1H, A2H) are precomputed on construction; the instance is
    treated as immutable afterwards.
    """

    A1: Any
    A2: Any
    s0: np.ndarray
    s1: np.ndarray
    c0: float
    A0: Optional[Any] = None
    A1H: Any = field(init=False, repr=False)
    A2H: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute adjoints; normalize s0/s1/c0."""
        self.A1H = _adjoint(self.A1)
        self.A2H = _adjoint(self.A2)
        self.s0 = np.asarray(self.s0, dtype=complex)
        self.s1 = np.asarray(self.s1, dtype=complex)
        self.c0 = float(self.c0)

    @property
    def n(self) -> int:
        """Dimension of the optimization variable x."""
        return self.A2.shape[1]

    def matvec_with_diagonal(
        self, d: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return x -> A x for A = A0 + Sym(A1 diag(d) A2), matrix-free."""
        d_conj = d.conj()

        if self.A0 is None:

            def matvec(x: np.ndarray) -> np.ndarray:
                return 0.5 * (
                    self.A1 @ (d * (self.A2 @ x))
                    + self.A2H @ (d_conj * (self.A1H @ x))
                )
        else:

            def matvec(x: np.ndarray) -> np.ndarray:
                return self.A0 @ x + 0.5 * (
                    self.A1 @ (d * (self.A2 @ x))
                    + self.A2H @ (d_conj * (self.A1H @ x))
                )

        return matvec

    def fs_columns(self, Pdiags: np.ndarray) -> np.ndarray:
        """Linear constraint terms: column j = A2^† (conj(P_j) ⊙ s1)."""
        return self.A2H @ (Pdiags.conj() * self.s1[:, None])
