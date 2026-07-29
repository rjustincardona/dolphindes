"""
Generalizable numerical utilities: conjugate gradients and inverse iteration.

Nothing in this module knows about QCQPs, barriers, or design problems — it
operates on Hermitian linear operators given as matvec callables.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

Matvec = Callable[[np.ndarray], np.ndarray]


class IndefiniteError(RuntimeError):
    """An iterative solve certified that the operator is not PSD.

    Conjugate gradients on a Hermitian operator produces a direction p with
    p^† A p < 0 (beyond rounding noise) only if A is indefinite. The
    normalized negative-curvature direction is carried when available: it is
    both a certificate and a useful penalty direction.
    """

    direction: Optional[np.ndarray] = None


def crdot(a: np.ndarray, b: np.ndarray) -> float:
    """Real-field inner product Re<a, b> (complex vectors as real 2n-vectors)."""
    return float(np.real(np.vdot(a, b)))


class CGSolver:
    """Conjugate gradients for Hermitian PSD systems, with a warm-start cache.

    The cache holds up to `CACHE_SLOTS` (b/|b|, x/|b|) pairs, evicted by true
    LRU (least-recently-USED, not least-recently-inserted): a cache hit moves
    its entry to the end of the list, so a right-hand-side "channel" that is
    reused every call (e.g. the primal xstar solve) stays resident regardless
    of how many *other* distinct channels (barrier-gradient solve, one per
    active fake source, ...) cycle through the remaining slots. A new
    right-hand side is matched by direction (cosine above `WARM_MIN_COSINE`)
    and the cached solution is phase/scale-corrected. A warm guess whose
    initial residual exceeds |b| falls back to the zero start, so
    warm-starting can only cost one extra matvec. The cache persists for the
    solver's lifetime — sequences of nearby systems (e.g. line-search trials)
    benefit the most. `CACHE_SLOTS=12` comfortably covers the common case of
    xstar + the barrier-gradient solve + up to 5 active fake sources (7
    distinct channels) with headroom.

    Negative curvature (p^† A p <= 0) beyond the rounding noise floor
    1e-12*|p||Ap| raises IndefiniteError carrying the direction; at-noise
    events are counted as breakdown and return the current iterate.
    Non-convergence at the iteration cap warns and returns the iterate.
    """

    CACHE_SLOTS = 12
    WARM_MIN_COSINE = 0.1

    def __init__(
        self,
        rtol: float = 1e-6,
        maxiter: Optional[int] = None,
        warm_start: bool = True,
    ) -> None:
        self.rtol = float(rtol)
        self.maxiter = maxiter
        self.warm_start = bool(warm_start)
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        self.stats: dict[str, int] = {
            "cg_solves": 0,
            "cg_iters": 0,
            "cg_indefinite": 0,
            "cg_noconv": 0,
            "cg_warm_hits": 0,
            "cg_breakdown": 0,
        }

    def _warm_lookup(self, b: np.ndarray, b_norm: float):
        best, best_cos, best_idx = None, self.WARM_MIN_COSINE, -1
        for idx, (b_unit, x_scaled) in enumerate(self._cache):
            overlap = np.vdot(b_unit, b)
            cos = abs(overlap) / b_norm
            if cos > best_cos:
                best, best_cos, best_idx = x_scaled * overlap, cos, idx
        return best, best_idx

    def _warm_store(self, b, b_norm, x, slot) -> None:
        entry = (b / b_norm, x / b_norm)
        if slot >= 0:
            # LRU: a hit is refreshed at the END of the list (most-recently
            # used), not overwritten in place -- otherwise a frequently-reused
            # channel (e.g. xstar) could be evicted purely because it happened
            # to be inserted before other, less-reused channels.
            del self._cache[slot]
        self._cache.append(entry)
        if len(self._cache) > self.CACHE_SLOTS:
            self._cache.pop(0)

    def solve(
        self, matvec: Matvec, b: np.ndarray, precond: Optional[Matvec] = None
    ) -> np.ndarray:
        """Solve A x = b for Hermitian PSD A given as a matvec callable.

        precond, if given, approximates z = M^{-1} r for some SPD M and
        turns this into preconditioned CG (the recurrence tracks
        rz = Re<r, z> in place of rs = Re<r, r>; convergence is still
        checked on the true residual rs). With precond=None, z == r and
        rz == rs identically at every step, so this reduces exactly to the
        original unpreconditioned recurrence -- bitwise unchanged.
        """
        n = b.shape[0]
        maxiter = self.maxiter if self.maxiter is not None else 10 * n

        b_norm = np.linalg.norm(b)
        if not np.isfinite(b_norm):
            raise ValueError("CG right-hand side contains non-finite entries.")
        if b_norm == 0.0:
            return np.zeros(n, dtype=complex)
        tol2 = (self.rtol * b_norm) ** 2

        slot = -1
        x = None
        if self.warm_start:
            x0, slot = self._warm_lookup(b, b_norm)
            if x0 is not None:
                r = b - matvec(x0)
                rs = np.real(np.vdot(r, r))
                if rs < b_norm**2:
                    x = x0
                    self.stats["cg_warm_hits"] += 1
        if x is None:
            x = np.zeros(n, dtype=complex)
            r = b.astype(complex, copy=True)
            rs = np.real(np.vdot(r, r))

        z = precond(r) if precond is not None else r
        rz = np.real(np.vdot(r, z)) if precond is not None else rs
        p = z.copy()
        rs_new = rs  # only reached by the closing warning if maxiter == 0
        self.stats["cg_solves"] += 1

        if rs <= tol2:
            if self.warm_start:
                self._warm_store(b, b_norm, x, slot)
            return x

        for it in range(maxiter):
            Ap = matvec(p)
            pAp = np.real(np.vdot(p, Ap))
            if pAp <= 0:
                # distinguish genuine negative curvature from roundoff-level
                # cancellation on an ill-conditioned PSD system
                noise = 1e-12 * np.linalg.norm(p) * np.linalg.norm(Ap)
                if pAp < -noise:
                    self.stats["cg_indefinite"] += 1
                    err = IndefiniteError(
                        f"CG negative curvature (p^†Ap = {pAp:.3e}) at "
                        f"iteration {it}: operator is not PSD."
                    )
                    err.direction = p / np.linalg.norm(p)
                    raise err
                self.stats["cg_breakdown"] += 1
                self.stats["cg_iters"] += it
                return x
            alpha = rz / pAp
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.real(np.vdot(r, r))
            if rs_new <= tol2:
                self.stats["cg_iters"] += it + 1
                if self.warm_start:
                    self._warm_store(b, b_norm, x, slot)
                return x
            z_new = precond(r) if precond is not None else r
            rz_new = np.real(np.vdot(r, z_new)) if precond is not None else rs_new
            p = z_new + (rz_new / rz) * p
            rz = rz_new

        self.stats["cg_iters"] += maxiter
        self.stats["cg_noconv"] += 1
        warnings.warn(
            f"CG did not reach rtol {self.rtol:.1e} in {maxiter} iterations "
            f"(relative residual {np.sqrt(rs_new) / b_norm:.2e}); returning "
            "the current iterate.",
            RuntimeWarning,
        )
        return x


def smallest_eigenpair(
    matvec: Matvec,
    solve: Callable[[np.ndarray], np.ndarray],
    n: int,
    v0: Optional[np.ndarray] = None,
    max_outer: int = 8,
    align_tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Approximate smallest eigenpair via inverse iteration v <- A^{-1}v.

    `solve` applies A^{-1} (e.g. a CGSolver bound to the same matvec) and may
    raise IndefiniteError; the certified negative-curvature direction is then
    returned as the eigenvector approximation (with its negative Rayleigh
    quotient). Pass the previous eigenvector as v0 to warm-start — it drifts
    slowly along optimization paths, typically leaving 1-3 outer iterations.

    Returns (v, Rayleigh quotient v^† A v) with unit-norm v.
    """
    if v0 is not None and v0.shape[0] == n:
        v = v0.copy()
    else:
        v = np.random.standard_normal(n) + 1j * np.random.standard_normal(n)
    v /= np.linalg.norm(v)

    for _ in range(max_outer):
        try:
            w = solve(v)
        except IndefiniteError as err:
            if err.direction is not None:
                v = err.direction
            return v, float(np.real(np.vdot(v, matvec(v))))
        w /= np.linalg.norm(w)
        aligned = abs(np.vdot(w, v)) > 1 - align_tol
        v = w
        if aligned:
            break

    return v, float(np.real(np.vdot(v, matvec(v))))
