"""
Matrix-free QCQP dual bounds: the log-barrier dual function and its BFGS solver.

This module owns two things:

- `_MatrixFreeQCQPImpl`, a near-verbatim port of the validated `slim` package's
  QCQP engine: the barrier-augmented dual (value and gradient, matrix-free via
  conjugate gradients), dual feasibility, and the BFGS solver. Constraint
  management and the annealed GCD pipeline live in `.gcd`; the generic
  numerics (CG with warm-start cache, inverse iteration, indefiniteness
  certificates) live in `.numerics`; the problem data and matrix-free operator
  kernels in `.problem`. Nothing here is ever assembled or factorized.
- `MatrixFreeSharedProjQCQP`, a thin adapter exposing the same public surface
  as `dolphindes.cvxopt.qcqp.SparseSharedProjQCQP`/`DenseSharedProjQCQP`
  (`solve_current_dual_problem`, `run_gcd`, `is_dual_feasible`, `get_dual`,
  `.current_xstar`/`.current_lags`/`.current_dual`/`.current_grad`,
  `.A0`/`.A1`/`.A2`/`.s0`/`.s1`/`.c0`) so it can be dropped into
  `Photonics_FDFD.setup_QCQP` alongside the existing Cholesky-based solvers,
  selected via the `matrix_free=True` constructor flag. It does NOT inherit
  from `_SharedProjQCQP`: that base class's `__init__` is entirely
  Cholesky-shaped (CSC coercion, `precomputed_As`/`Fs`, a CHOLMOD/`cho_factor`
  factorization cache) with no matrix-free equivalent, and two of its abstract
  methods (`_factorize`, `_Acho_solve`) have no matrix-free analogue either.

A log barrier -w * log(g_q + c) on the violation g_q of the (protected)
constraint q keeps the optimization away from the PSD boundary; w is annealed
by the GCD pipeline (see `.gcd`). Fake sources use the decayed end-of-GCD
schedule: one persistent source (the smallest eigenvector of A(lags), shared
with GCD's own constraint-generation step) is appended after every GCD
iteration's BFGS solve, scaled to contribute ~source_ratio of the current
dual; the active list is FIFO-capped at max_sources and injected into every
dual() evaluation automatically, decaying at each annealing-round boundary.

The numerics deliberately replicate validated `slim`/dolphindes behavior,
including a few subtle, bound-affecting details; see the inline comments
marked "replicates".
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from dolphindes.util import Projectors

from .numerics import CGSolver, IndefiniteError, smallest_eigenpair
from .problem import DesignProblem

if TYPE_CHECKING:
    from .gcd import MatrixFreeGCDParams

__all__ = [
    "MatrixFreeSharedProjQCQP",
    "MatrixFreeBFGSParams",
    "MatrixFreeDualResult",
    "DesignProblem",
    "IndefiniteError",
]


@dataclass
class MatrixFreeBFGSParams:
    """BFGS solver settings (defaults tuned for GCD-internal solves)."""

    opttol: float = 1e-2
    gradConverge: bool = False
    min_inner_iter: int = 5
    max_restart: float = 1
    break_iter_period: int = 20
    verbose: int = 0


MatrixFreeDualResult = namedtuple(
    "MatrixFreeDualResult",
    ["value", "grad", "raw_value", "raw_grad", "barrier_value", "penalty_value"],
)
# value = raw_value + barrier_value + penalty_value: what BFGS minimizes.
# raw_value is the reportable dual bound (valid whenever A(lags) is PSD).


class _MatrixFreeQCQPImpl:
    """Diagonal shared-projection QCQP with log-barrier dual optimization.

    Parameters
    ----------
    problem : DesignProblem
        The quadratic program data (see .problem).
    Pdiags : (n, k) complex ndarray
        Columns are the diagonals of the projector constraints. Column
        barrier_idx must have PSD Sym(A1 P A2) (by convention index 1).
    barrier_weight, barrier_shift, barrier_idx
        Log barrier -w log(g_idx + shift); the weight is annealed in place by
        the GCD pipeline. Must satisfy barrier_idx < n_protected.
    n_protected : int
        Leading projector constraints never modified by GCD merging or
        orthonormalization (keeps the barrier constraint's identity fixed).
    max_sources, source_ratio, source_decay : persistent fake-source schedule
        One source is appended per GCD iteration (add_source), FIFO-capped at
        max_sources; each new source is scaled to contribute ~source_ratio of
        the current dual. decay_sources (called once per annealing round)
        shrinks every active source's contribution by source_decay and
        derates the ratio for new sources equally. Each active source costs
        one extra CG solve per dual evaluation.

    Every CG solve is unpreconditioned: A0/A1/A2 may be any linear operator
    (see DesignProblem), not necessarily a sparse matrix with an accessible
    diagonal, so there is no general-purpose preconditioner here.
    """

    def __init__(
        self,
        problem: DesignProblem,
        Pdiags: np.ndarray,
        barrier_weight: float = 1e-3,
        barrier_shift: float = 1e-2,
        barrier_idx: int = 1,
        n_protected: int = 2,
        max_sources: int = 5,
        source_ratio: float = 1e-2,
        source_decay: float = 0.1,
        cg_rtol: float = 1e-6,
        cg_maxiter: Optional[int] = None,
        cg_warm_start: bool = True,
        verbose: int = 0,
    ) -> None:
        self.prob = problem
        self.n = problem.n

        self.Pdiags = np.array(Pdiags, dtype=complex, copy=True)
        assert self.Pdiags.shape[0] == self.n
        assert self.Pdiags.shape[1] > barrier_idx, "barrier constraint missing"
        self.Fs = problem.fs_columns(self.Pdiags)

        if not 0 < barrier_weight:
            raise ValueError("barrier_weight must be positive")
        if barrier_shift < 0:
            raise ValueError("barrier_shift must be non-negative")
        if not barrier_idx < n_protected <= self.k:
            raise ValueError("need barrier_idx < n_protected <= #constraints")
        self.barrier_weight = float(barrier_weight)
        self.barrier_shift = float(barrier_shift)
        self.barrier_idx = int(barrier_idx)
        self.n_protected = int(n_protected)
        self.max_sources = int(max_sources)
        self.source_ratio = float(source_ratio)
        self.source_decay = float(source_decay)
        self.sources: list[np.ndarray] = []

        # the CG solver's warm-start cache deliberately persists across lags
        # changes, GCD iterations, and anneal rounds (replicates dolphindes)
        self.cg = CGSolver(rtol=cg_rtol, maxiter=cg_maxiter, warm_start=cg_warm_start)
        # inverse-iteration eigenvector warm start (persists likewise)
        self._psd_penalty_v: Optional[np.ndarray] = None
        # cached orthonormal real-embedded basis for the protected Pdiags
        # columns (.gcd.add_constraints): never invalidated because
        # protected columns are immutable after construction
        self._prot_basis: Optional[np.ndarray] = None

        self.verbose = int(verbose)
        self.current_lags: Optional[np.ndarray] = None
        self.current_dual: Optional[float] = None
        self.current_grad: Optional[np.ndarray] = None
        self.current_xstar: Optional[np.ndarray] = None
        self.last_gcd_rescue_depths: list[int] = []
        # count of line-search phase-2 loops cut off by _MAX_IMPROVE_SHRINKS;
        # should be 0 in any healthy run (see _line_search) — a nonzero value
        # means a stall was caught, worth investigating
        self.line_search_stalls = 0

    # ------------------------------------------------------------------
    # problem-operator shorthands
    # ------------------------------------------------------------------

    @property
    def k(self) -> int:
        """Number of projector constraints (= columns of Pdiags)."""
        return self.Pdiags.shape[1]

    @property
    def solve_stats(self) -> dict[str, int]:
        """Iterative-solver counters (delegated to the CG solver)."""
        return self.cg.stats

    def _operator(self, lags: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Matvec for A(lags), i.e. x -> A(lags) x, sharing one d = Pdiags @ lags."""
        d = self.Pdiags @ lags
        return self.prob.matvec_with_diagonal(d)

    def _S(self, lags: np.ndarray) -> np.ndarray:
        """Linear term of the Lagrangian: S = s0 + Fs @ lags."""
        return self.prob.s0 + self.Fs @ lags

    # ------------------------------------------------------------------
    # dual function
    # ------------------------------------------------------------------

    def dual(
        self,
        lags: np.ndarray,
        grad: bool = False,
        penalty_vectors: Optional[list[np.ndarray]] = None,
    ) -> MatrixFreeDualResult:
        """Barrier-augmented dual value (and gradient) at lags, matrix-free.

        value = raw + barrier + penalties, where the penalty vectors are
        self.sources (the persistent fake-source list, always included) plus
        any extra penalty_vectors passed in. An evaluation at a not-PSD point
        (CG negative-curvature certificate) is reported as infinitely bad;
        indefiniteness surfacing in the auxiliary solves (barrier gradient,
        fake sources) poisons only the value, leaving the raw gradient finite
        so the optimizer can walk away (replicates the dolphindes rules).
        """
        pv = list(penalty_vectors) if penalty_vectors else []
        pv += self.sources
        p = self.prob
        matvec = self._operator(lags)
        try:
            xstar = self.cg.solve(matvec, self._S(lags))
        except (IndefiniteError, ValueError):
            # ValueError: non-finite RHS, i.e. lags themselves are garbage
            # (e.g. an overflowed trial point) — infinitely bad, not fatal
            return MatrixFreeDualResult(np.inf, None, np.inf, None, np.inf, 0.0)
        # replicates: xAx via an explicit matvec, not Re(x^†S) — the two
        # differ at CG-tolerance level and that difference feeds stall logic
        raw_value = np.real(np.vdot(xstar, matvec(xstar))) + p.c0

        y = p.A2 @ xstar
        u = p.A1H @ xstar
        w_vec = u.conj() * y

        raw_grad = None
        if grad:
            raw_grad = -np.real(w_vec @ self.Pdiags) + 2 * np.real(
                xstar.conj() @ self.Fs
            )

        q = self.barrier_idx
        gq = float(
            -np.real(w_vec @ self.Pdiags[:, q])
            + 2 * np.real(np.vdot(xstar, self.Fs[:, q]))
        )
        barrier_arg = gq + self.barrier_shift
        barrier_grad = None
        if barrier_arg <= 0:
            barrier_value = np.inf
        else:
            barrier_value = -self.barrier_weight * np.log(barrier_arg)
            if grad:
                # d g_q / d lags = row q of the dual Hessian
                # = 2 Re[(A^{-1} Ftot_q)^† Ftot], Ftot = Fs - [A_k x*]_k
                Ftot = self.Fs - 0.5 * (
                    p.A1 @ (self.Pdiags * y[:, None])
                    + p.A2H @ (self.Pdiags.conj() * u[:, None])
                )
                try:
                    z = self.cg.solve(matvec, np.ascontiguousarray(Ftot[:, q]))
                    gq_grad = 2 * np.real(z.conj() @ Ftot)
                    barrier_grad = -(self.barrier_weight / barrier_arg) * gq_grad
                except IndefiniteError:
                    barrier_value = np.inf

        penalty_value = 0.0
        penalty_grad = None
        if pv:
            try:
                p_solved = [self.cg.solve(matvec, v) for v in pv]
                penalty_value = sum(
                    np.real(np.vdot(pj, vj))
                    for pj, vj in zip(p_solved, pv)
                )
                if grad:
                    penalty_grad = np.zeros(self.k)
                    for pj in p_solved:
                        wj = (p.A1H @ pj).conj() * (p.A2 @ pj)
                        penalty_grad += -np.real(wj @ self.Pdiags)
            except IndefiniteError:
                penalty_value = np.inf
                penalty_grad = None

        value = raw_value + barrier_value + penalty_value
        total_grad = None
        if grad and raw_grad is not None:
            total_grad = raw_grad.copy()
            if barrier_grad is not None:
                total_grad += barrier_grad
            if penalty_grad is not None:
                total_grad += penalty_grad

        return MatrixFreeDualResult(
            value, total_grad, raw_value, raw_grad, barrier_value, penalty_value
        )

    def constraint_violation(self, lags: np.ndarray, q: Optional[int] = None) -> float:
        """Violation g_q at the dual maximizer x*(lags); one CG solve.

        Raises IndefiniteError if A(lags) is certified not PSD.
        """
        if q is None:
            q = self.barrier_idx
        p = self.prob
        matvec = self._operator(lags)
        xstar = self.cg.solve(matvec, self._S(lags))
        w_vec = (p.A1H @ xstar).conj() * (p.A2 @ xstar)
        return float(
            -np.real(w_vec @ self.Pdiags[:, q])
            + 2 * np.real(np.vdot(xstar, self.Fs[:, q]))
        )

    def _raw_and_barrier_from_xstar(
        self, lags: np.ndarray, xstar: np.ndarray
    ) -> tuple[float, float, float]:
        """(raw_value, barrier_value, g_q) from an already-solved x*.

        Mirrors dual()'s and constraint_violation()'s formulas exactly but
        takes xstar directly instead of re-solving A(lags) x = S(lags) --
        for callers that already know x* is the solution at these lags
        (e.g. right after bfgs() sets current_lags/current_xstar together;
        see .gcd.run_gcd and add_source).
        """
        p = self.prob
        matvec = p.matvec_with_diagonal(self.Pdiags @ lags)
        raw_value = np.real(np.vdot(xstar, matvec(xstar))) + p.c0
        y = p.A2 @ xstar
        u = p.A1H @ xstar
        w_vec = u.conj() * y
        q = self.barrier_idx
        gq = float(
            -np.real(w_vec @ self.Pdiags[:, q])
            + 2 * np.real(np.vdot(xstar, self.Fs[:, q]))
        )
        barrier_arg = gq + self.barrier_shift
        barrier_value = (
            np.inf if barrier_arg <= 0 else -self.barrier_weight * np.log(barrier_arg)
        )
        return raw_value, barrier_value, gq

    # ------------------------------------------------------------------
    # feasibility
    # ------------------------------------------------------------------

    def is_dual_feasible(self, lags: np.ndarray) -> bool:
        """Barrier-domain feasibility: g_q + shift > 0 via one CG solve.

        The domain lies strictly inside the PSD cone (g_q -> -inf at the
        boundary); a trial past both walls is still rejected because CG
        certifies indefiniteness.
        """
        try:
            gq = self.constraint_violation(lags)
        except (IndefiniteError, ValueError):
            return False
        return bool(gq + self.barrier_shift > 0)

    def _feasible_start_ok(self, lags: np.ndarray) -> bool:
        """Require g_q > 0 strictly, a stricter condition for starting points."""
        try:
            gq = self.constraint_violation(lags)
        except (IndefiniteError, ValueError):
            return False
        return bool(gq > 0)

    def find_feasible_lags(self, start: float = 0.1, limit: float = 1e8) -> np.ndarray:
        """Scan lags[1] upward until g_1 > 0 (replicates dolphindes).

        Each attempt costs one CG solve (via _feasible_start_ok), and the
        worst case is ~52 attempts (start=0.1, limit=1e8, factor 1.5) -- with
        no per-attempt output this can look like a silent hang on a slow
        (e.g. unpreconditioned, ill-conditioned) operator, so print progress
        per attempt when verbose.
        """
        if self.current_lags is not None and self._feasible_start_ok(
            self.current_lags
        ):
            return self.current_lags
        init_lags = np.random.random(self.k) * 1e-6
        init_lags[1] = start
        attempt = 0
        while not self._feasible_start_ok(init_lags):
            attempt += 1
            if self.verbose > 0:
                print(
                    f"  feasible-lags search: attempt {attempt}, "
                    f"lags[1] = {init_lags[1]:.4g} infeasible, trying higher"
                )
            init_lags[1] *= 1.5
            if init_lags[1] > limit:
                raise ValueError("Could not find a feasible dual starting point.")
        if self.verbose > 0:
            print(f"Found feasible starting lags with lags[1] = {init_lags[1]:.4g}")
        return init_lags

    # ------------------------------------------------------------------
    # smallest eigenpair (fake sources + GCD constraint generation)
    # ------------------------------------------------------------------

    def _psd_penalty(self, lags: np.ndarray) -> tuple[np.ndarray, float]:
        """Approximate smallest eigenpair of A(lags), warm-started."""
        matvec = self._operator(lags)
        v0 = self._psd_penalty_v
        if v0 is not None and not np.all(np.isfinite(v0)):
            v0 = None  # drop a warm start poisoned by a derailed evaluation
        v, lam = smallest_eigenpair(
            matvec,
            solve=lambda b: self.cg.solve(matvec, b),
            n=self.n,
            v0=v0,
        )
        self._psd_penalty_v = v
        return v, lam

    def add_source(self, v: np.ndarray) -> bool:
        """Append a persistent fake source at current_lags using eigenvector v.

        v is expected to be the smallest eigenvector of A(current_lags) --
        shared with GCD's own constraint-generation step rather than solved
        twice; see .gcd.gcd_round, which calls this once per GCD iteration
        right after bfgs() converges. Scales v so its dual contribution is
        ~source_ratio of the current dual (reusing current_xstar via
        _raw_and_barrier_from_xstar, no extra xstar solve), FIFO-caps the
        active list at max_sources. Silently skips on any non-finite/
        indefinite outcome -- a missing source never hurts correctness, only
        boundary repulsion.
        """
        if self.current_lags is None or self.current_xstar is None:
            return False
        if self.current_dual is None or not np.isfinite(self.current_dual):
            return False
        raw, barrier, _ = self._raw_and_barrier_from_xstar(
            self.current_lags, self.current_xstar
        )
        if not np.isfinite(barrier):
            return False
        matvec = self._operator(self.current_lags)
        try:
            solved = self.cg.solve(matvec, v)
        except (IndefiniteError, ValueError):
            return False
        pen_val = raw + barrier + np.real(np.vdot(solved, v))
        epsS = np.sqrt(self.source_ratio * np.abs(self.current_dual / pen_val))
        if not (np.isfinite(epsS) and np.all(np.isfinite(v))):
            return False
        self.sources.append(epsS * v)
        if len(self.sources) > self.max_sources:
            self.sources.pop(0)
        return True

    def decay_sources(self) -> None:
        """Decay every persistent source's dual contribution by source_decay.

        Called once per annealing-round boundary (.gcd.run_gcd). The
        contribution v^H A^-1 v is quadratic in the vector, so vectors scale
        by sqrt(source_decay); new sources' ratio decays equally so the
        schedule keeps pace with the annealing barrier weight.
        """
        scale = np.sqrt(self.source_decay)
        self.sources = [scale * v for v in self.sources]
        self.source_ratio *= self.source_decay

    # ------------------------------------------------------------------
    # BFGS dual solver
    # ------------------------------------------------------------------

    # Relative margin an evaluation must beat the running best by to count as
    # real improvement (below this, a "decrease" is CG/BLAS noise, not
    # progress — see _line_search).
    _IMPROVE_NOISE_FLOOR = 1e-12
    # Hard cap on phase-2 shrink steps. Every shrink multiplies alpha by
    # c_reduct=0.7, so alpha underflows to exactly 0.0 (making the trial
    # point bit-identical to x0) within ~4100 steps even starting from the
    # top of the float64 range — a genuine monotonic descent can therefore
    # never need more than that many steps. This cap is set well above that
    # bound purely as a backstop against a loop kept alive by evaluation
    # noise rather than real improvement (observed in the source project: a
    # stalled run once spent 28 hours and ~600M CG solves re-evaluating a
    # converged point because ULP-level noise between nominally-identical
    # warm-started CG solves kept registering as "still decreasing").
    _MAX_IMPROVE_SHRINKS = 10_000

    def _line_search(self, direction, x0, init_step):
        """Two-phase backtracking line search (replicates dolphindes).

        Phase 1 backtracks on feasibility (cap 120 -> zero step). Phase 2
        evaluates values, treating non-finite values like infeasibility
        (cap 60 shrinks), continuing while improving by more than a noise
        floor (cap _MAX_IMPROVE_SHRINKS shrinks, never reached by genuine
        descent — see _MAX_IMPROVE_SHRINKS).
        """
        c_reduct = 0.7
        alpha = init_step

        feas_backtracks = 0
        while not self.is_dual_feasible(x0 + alpha * direction):
            alpha *= c_reduct
            feas_backtracks += 1
            if feas_backtracks > 120:
                # x0 itself flips infeasible under the stochastic (iterative)
                # feasibility check; give up on this direction
                return 0.0

        opt_val = np.inf
        alpha_opt = alpha
        nonfinite_shrinks = 0
        improve_shrinks = 0
        while True:
            tmp = self.dual(x0 + alpha * direction, grad=False).value
            if not np.isfinite(tmp):
                nonfinite_shrinks += 1
                if nonfinite_shrinks > 60:
                    break
                alpha *= c_reduct
                continue
            if not np.isfinite(opt_val) or (
                tmp < opt_val - self._IMPROVE_NOISE_FLOOR * abs(opt_val)
            ):
                opt_val = tmp
                alpha_opt = alpha
            else:
                break
            alpha *= c_reduct
            improve_shrinks += 1
            if improve_shrinks > self._MAX_IMPROVE_SHRINKS:
                self.line_search_stalls += 1
                break

        return alpha_opt

    def bfgs(
        self,
        init_lags: Optional[np.ndarray] = None,
        params: Optional[MatrixFreeBFGSParams] = None,
    ) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
        """Minimize the barrier-augmented dual with BFGS (replicates dolphindes).

        Sets current_lags / current_dual / current_grad / current_xstar and
        returns (dual, lags, grad). Returns the best finite iterate if the
        run derails.
        """
        p = params or MatrixFreeBFGSParams()
        if init_lags is None:
            init_lags = self.find_feasible_lags()
        x = np.array(init_lags, dtype=float)
        ndof = x.size

        outer_iter = 0
        best_fx, best_x, best_grad = None, None, None
        prev_fx_outer = np.inf

        while True:  # outer loop: restarts if not yet converged
            res = self.dual(x, grad=True)
            fx, xgrad = res.value, res.grad

            last_step = 1.0
            Hinv = np.eye(ndof)
            inner_iter = 0
            consecutive_bad = 0
            prev_fx = np.inf

            if xgrad is None or not np.isfinite(fx):
                break  # unusable start; fall back to best-so-far

            while True:  # inner BFGS loop
                inner_iter += 1
                if self.verbose > 1:
                    print(f"  BFGS inner {inner_iter}, fx = {fx}")

                if np.isfinite(fx) and (best_fx is None or fx < best_fx):
                    best_fx, best_x, best_grad = fx, x.copy(), xgrad.copy()

                direction = -Hinv @ xgrad
                direction /= np.linalg.norm(direction)

                opt_step = self._line_search(direction, x, last_step)
                if np.isclose(opt_step, last_step, atol=0.0):
                    # no backtracking happened: try a more aggressive step
                    last_step = opt_step * 2
                else:
                    last_step = opt_step

                delta = opt_step * direction
                old_grad = xgrad.copy()
                x = x + delta

                res = self.dual(x, grad=True)
                new_fx, new_grad = res.value, res.grad

                if new_grad is None or not np.isfinite(new_fx):
                    # unusable accepted point: revert and retry cautiously
                    x = x - delta
                    Hinv = np.eye(ndof)
                    last_step = max(opt_step * 0.1, 1e-14)
                    consecutive_bad += 1
                    if consecutive_bad >= 3 or self._bfgs_break(
                        inner_iter, fx, prev_fx, x, xgrad, p
                    ):
                        break
                    continue
                consecutive_bad = 0

                gamma = new_grad - old_grad
                gd = gamma @ delta
                if gd == 0 or not np.isfinite(gd):
                    # degenerate step (e.g. zero step from a capped line
                    # search): 1/gd would poison Hinv with non-finite
                    # entries; restart the approximation instead
                    Hinv = np.eye(ndof)
                else:
                    rho = 1.0 / gd
                    ident = np.eye(ndof)
                    Hinv = (ident - rho * np.outer(delta, gamma)) @ Hinv @ (
                        ident - rho * np.outer(gamma, delta)
                    ) + rho * np.outer(delta, delta)

                fx, xgrad = new_fx, new_grad

                brk, prev_fx = self._bfgs_break_update(
                    inner_iter, fx, prev_fx, x, xgrad, p
                )
                if brk:
                    break

            if not np.isfinite(fx) or (
                abs(prev_fx_outer - fx) < abs(fx) * p.opttol
                or np.isclose(fx, 0, atol=1e-14)
                or outer_iter > p.max_restart
            ):
                break
            prev_fx_outer = fx
            outer_iter += 1

        if best_fx is not None and (not np.isfinite(fx) or best_fx < fx):
            x, fx, xgrad = best_x, best_fx, best_grad

        self.current_lags = x
        self.current_dual = fx
        self.current_grad = xgrad
        try:
            matvec = self._operator(x)
            self.current_xstar = self.cg.solve(matvec, self._S(x))
        except IndefiniteError:
            # the optimum can sit at the numerical PSD boundary; keep the
            # previous x* (it only seeds constraint generation)
            if self.verbose > 0:
                print("Warning: x* not computable at the returned optimum.")
        return fx, x, xgrad

    def _bfgs_break(self, inner_iter, fx, prev_fx, x, xgrad, p) -> bool:
        brk, _ = self._bfgs_break_update(inner_iter, fx, prev_fx, x, xgrad, p)
        return brk

    def _bfgs_break_update(self, inner_iter, fx, prev_fx, x, xgrad, p):
        """Inner-loop convergence checks (replicates the dolphindes tests)."""
        if inner_iter > p.min_inner_iter and xgrad is not None:
            fminus = fx - x @ xgrad
            remaining = np.abs(x) @ np.abs(xgrad)
            if p.gradConverge:
                if (
                    np.abs(fx - fminus) < p.opttol * np.abs(fminus)
                    and np.abs(remaining) < p.opttol * np.abs(fminus)
                    and np.linalg.norm(xgrad) < p.opttol * np.abs(fx)
                ):
                    return True, prev_fx
            elif np.abs(fx - fminus) < p.opttol * np.abs(fminus) and np.abs(
                remaining
            ) < p.opttol * np.abs(fminus):
                return True, prev_fx
        if inner_iter % p.break_iter_period == 0:
            if np.abs(prev_fx - fx) < np.abs(fx) * p.opttol or np.isclose(
                fx, 0, atol=1e-14
            ):
                return True, prev_fx
            return False, fx
        return False, prev_fx

    # ------------------------------------------------------------------
    # GCD pipeline (implemented in .gcd; thin convenience wrapper)
    # ------------------------------------------------------------------

    def gcd(self, params: Optional["MatrixFreeGCDParams"] = None) -> dict:
        """Run the full annealed GCD pipeline (see .gcd.run_gcd)."""
        from .gcd import run_gcd

        return run_gcd(self, params)


def _default_pstruct(Plist: list) -> Any:
    """Superset sparsity structure of Plist (mirrors _SharedProjQCQP's default).

    Duplicated here (rather than imported) because it is inline logic inside
    dolphindes.cvxopt._base_qcqp.__init__, not a standalone reusable function
    -- this keeps dolphindes/cvxopt/_base_qcqp.py completely untouched.
    """
    P0 = Plist[0]
    Pstruct = (
        P0.astype(complex).copy()
        if sp.issparse(P0)
        else np.asarray(P0, dtype=complex).copy()
    )
    for P in Plist:
        coef = (np.random.rand() + 0.01) + 0j
        Pcomplex = P.astype(complex) if sp.issparse(P) else np.asarray(P, dtype=complex)
        Pstruct += coef * Pcomplex
    return sp.csc_array(Pstruct, dtype=complex)


class MatrixFreeSharedProjQCQP:
    """Matrix-free (CG-based, log-barrier) alternative to SharedProjQCQP.

    Constructed with the same canonical argument order as
    `dolphindes.cvxopt._base_qcqp._SharedProjQCQP` --
    `(A0, s0, c0, A1, A2, s1, Plist, Pstruct=None, B_j=None, s_2j=None,
    c_2j=None, verbose=0)` -- so `Photonics_FDFD.setup_QCQP` can instantiate
    it in place of `SparseSharedProjQCQP`/`DenseSharedProjQCQP` with no other
    changes. Diagonal projectors only (raises ValueError otherwise -- every
    real call site in `setup_QCQP` only ever builds diagonal projectors); no
    general constraints (B_j/s_2j/c_2j must be empty -- raises
    NotImplementedError otherwise, since the matrix-free engine has no
    analogue for them).

    Extra matrix-free-specific tuning knobs (barrier_weight, cg_rtol, etc.,
    see `_MatrixFreeQCQPImpl`) are accepted as keyword-only arguments with the
    same defaults as the underlying engine; `setup_QCQP` does not pass any of
    these today, so the defaults are what's used in practice.
    """

    def __init__(
        self,
        A0,
        s0: np.ndarray,
        c0: float,
        A1,
        A2,
        s1: np.ndarray,
        Plist: list,
        Pstruct=None,
        B_j: Optional[list] = None,
        s_2j: Optional[list] = None,
        c_2j: Optional[np.ndarray] = None,
        verbose: int = 0,
        **matrix_free_kwargs: Any,
    ) -> None:
        if B_j or s_2j or (c_2j is not None and len(c_2j) > 0):
            raise NotImplementedError(
                "MatrixFreeSharedProjQCQP supports only shared diagonal "
                "projector constraints; general constraints (B_j/s_2j/c_2j) "
                "have no matrix-free analogue."
            )
        if Pstruct is None:
            Pstruct = _default_pstruct(list(Plist))
        proj = Projectors(list(Plist), Pstruct)
        if not proj.is_diagonal():
            raise ValueError(
                "MatrixFreeSharedProjQCQP requires diagonal projector "
                "constraints (got general/non-diagonal projectors)."
            )

        problem = DesignProblem(A0=A0, A1=A1, A2=A2, s0=s0, s1=s1, c0=c0)
        self._impl = _MatrixFreeQCQPImpl(
            problem, proj.Pdiags, verbose=verbose, **matrix_free_kwargs
        )

    # ------------------------------------------------------------------
    # data-attribute parity with SparseSharedProjQCQP/DenseSharedProjQCQP
    # ------------------------------------------------------------------

    @property
    def A0(self):
        """Objective quadratic matrix (delegates to the wrapped DesignProblem)."""
        return self._impl.prob.A0

    @property
    def A1(self):
        """Constraint bilinear-form matrix A1 (delegates to DesignProblem)."""
        return self._impl.prob.A1

    @property
    def A2(self):
        """Constraint bilinear-form matrix A2 (delegates to DesignProblem)."""
        return self._impl.prob.A2

    @property
    def s0(self) -> np.ndarray:
        """Objective linear vector (delegates to DesignProblem)."""
        return self._impl.prob.s0

    @property
    def s1(self) -> np.ndarray:
        """Constraint linear vector (delegates to DesignProblem)."""
        return self._impl.prob.s1

    @property
    def c0(self) -> float:
        """Objective constant (delegates to DesignProblem)."""
        return self._impl.prob.c0

    @property
    def current_xstar(self) -> Optional[np.ndarray]:
        """Primal maximizer at the current lags, or None before solving."""
        return self._impl.current_xstar

    @property
    def current_lags(self) -> Optional[np.ndarray]:
        """Current Lagrange multipliers, or None before solving."""
        return self._impl.current_lags

    @property
    def current_dual(self) -> Optional[float]:
        """Current (barrier-augmented) dual value, or None before solving."""
        return self._impl.current_dual

    @property
    def current_grad(self) -> Optional[np.ndarray]:
        """Gradient at the current lags, or None before solving."""
        return self._impl.current_grad

    @property
    def solve_stats(self) -> dict:
        """Iterative-solver counters (delegated to the underlying CG solver)."""
        return self._impl.solve_stats

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------

    def is_dual_feasible(self, lags: np.ndarray) -> bool:
        """See `_MatrixFreeQCQPImpl.is_dual_feasible`."""
        return self._impl.is_dual_feasible(lags)

    def get_dual(
        self,
        lags: np.ndarray,
        get_grad: bool = False,
        get_hess: bool = False,
        penalty_vectors: Optional[list] = None,
    ) -> Tuple[float, Optional[np.ndarray], None, MatrixFreeDualResult]:
        """Evaluate the dual function, matching SharedProjQCQP.get_dual's shape.

        Returns (dualval, grad, hess, dual_aux) like upstream's own
        `get_dual` -- `hess` is always None (no Newton/Hessian path exists in
        the matrix-free engine) and `dual_aux` is a `MatrixFreeDualResult`
        namedtuple (a different shape from upstream's own DualAux, since the
        matrix-free engine tracks barrier/penalty terms that have no upstream
        counterpart) -- a documented, deliberate deviation limited to this
        diagnostic return slot.
        """
        if get_hess:
            raise NotImplementedError(
                "MatrixFreeSharedProjQCQP has no Hessian/Newton path."
            )
        res = self._impl.dual(lags, grad=get_grad, penalty_vectors=penalty_vectors)
        return res.value, res.grad, None, res

    def solve_current_dual_problem(
        self,
        method: str,
        opt_params: Optional[MatrixFreeBFGSParams] = None,
        init_lags: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray], None, Optional[np.ndarray]]:
        """Optimize the dual problem with BFGS (matches upstream's 5-tuple shape).

        Only `method="bfgs"` is supported (the matrix-free engine has no
        Newton path); any other value raises ValueError. Returns
        (current_dual, current_lags, current_grad, current_hess, current_xstar)
        like `_SharedProjQCQP.solve_current_dual_problem` -- `current_hess` is
        always None.
        """
        if method != "bfgs":
            raise ValueError(
                f"MatrixFreeSharedProjQCQP only supports method='bfgs' "
                f"(got {method!r}); it has no Newton/Hessian path."
            )
        fx, x, grad = self._impl.bfgs(init_lags=init_lags, params=opt_params)
        return fx, x, grad, None, self._impl.current_xstar

    def run_gcd(self, gcd_params: Optional["MatrixFreeGCDParams"] = None) -> dict:
        """Run the annealed GCD pipeline (see .gcd.run_gcd).

        Like `_SharedProjQCQP.run_gcd`, mutates this QCQP's constraints and
        current_lags/current_dual/current_xstar in place. Additionally
        returns the diagnostics dict the underlying pipeline already produces
        (bound, history, solve_stats, ...) -- purely additive; existing
        call sites that ignore the return value are unaffected.
        """
        return self._impl.gcd(gcd_params)

    def add_constraints(
        self, added_Pdata_list: list, orthonormalize: bool = True
    ) -> None:
        """See `.gcd.add_constraints` (parity with SharedProjQCQP's method)."""
        from .gcd import add_constraints

        add_constraints(self._impl, added_Pdata_list, orthonormalize=orthonormalize)

    def merge_lead_constraints(self, merged_num: int) -> None:
        """See `.gcd.merge_lead_constraints` (parity with SharedProjQCQP's method)."""
        from .gcd import merge_lead_constraints

        merge_lead_constraints(self._impl, merged_num)
