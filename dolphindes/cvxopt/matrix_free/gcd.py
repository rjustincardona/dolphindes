"""
Generalized constraint descent (GCD) and the annealed log-barrier pipeline.

GCD tightens a QCQP dual bound by iteratively adding shared-projection
constraints likely to help (the maximum-violation direction at the current
dual maximizer and the smallest-eigenvector direction of A(lags)) and merging
older ones to cap the constraint count. The first `n_protected` constraints
are never modified, which keeps the barrier constraint's identity fixed.

`run_gcd()` is the full pipeline: it anneals the barrier weight, runs a GCD
round per weight (each round calls the engine's `bfgs` as needed), and
reports the best bound among rounds that end inside the barrier domain (a
finite barrier value implies PSD-verified multipliers).

Fake sources follow the decayed end-of-GCD schedule: one persistent source is
appended after each GCD iteration's BFGS solve (`add_source`, reusing the
same smallest-eigenvector solve this module's own constraint generation
needs -- no duplicate eigensolve) and every active source decays at each
annealing step (`decay_sources`); see `.qcqp` for the schedule details.

Distinct from `dolphindes.cvxopt.gcd` (upstream's own GCD module, built
around Newton/Cholesky-style solving with no barrier or annealing concept) --
`MatrixFreeGCDParams` is a differently-named, differently-shaped dataclass
from `dolphindes.cvxopt.gcd.GCDHyperparameters` so the two are never
accidentally interchangeable.

The functions here mutate the passed engine (constraints, multipliers,
barrier weight); `_MatrixFreeQCQPImpl.gcd()` /
`MatrixFreeSharedProjQCQP.run_gcd()` are thin convenience wrappers around
`run_gcd()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.linalg as la

from .numerics import IndefiniteError, crdot
from .qcqp import MatrixFreeBFGSParams

if TYPE_CHECKING:
    from .qcqp import _MatrixFreeQCQPImpl


@dataclass
class MatrixFreeGCDParams:
    """GCD + annealing pipeline settings for the matrix-free engine."""

    max_cstrt_num: int = 10
    max_gcd_iter_num: int = 50
    gcd_iter_period: int = 1
    gcd_tol: float = 1e-2
    stall_confirm_iters: int = 3
    orthonormalize: bool = True
    anneal_factor: float = 1e-2
    final_weight: float = 1e-8
    opt_params: Optional[MatrixFreeBFGSParams] = None


# ----------------------------------------------------------------------
# constraint management
# ----------------------------------------------------------------------


def add_constraints(
    qcqp: "_MatrixFreeQCQPImpl",
    new_cols: list[np.ndarray],
    orthonormalize: bool = True,
) -> None:
    """Append projector constraints (diagonals), zero initial multipliers.

    New columns are Gram-Schmidt orthogonalized against the span of the
    protected columns (via an auxiliary orthonormal basis; the protected
    columns themselves are never modified) and against the existing
    orthonormal unprotected columns, in order.
    """
    new_cols = [np.array(c, dtype=complex, copy=True) for c in new_cols]
    if qcqp.current_lags is not None:
        qcqp.current_lags = np.append(qcqp.current_lags, np.zeros(len(new_cols)))

    if orthonormalize:
        npr = qcqp.n_protected
        if qcqp._prot_basis is None:
            # protected Pdiags columns never change after construction, so
            # this QR is computed once and cached on the QCQP instance
            prot = qcqp.Pdiags[:, :npr]
            realext = np.vstack((np.real(prot), np.imag(prot)))
            prot_q, _ = la.qr(realext, mode="economic")
            qcqp._prot_basis = prot_q[: qcqp.n, :] + 1j * prot_q[qcqp.n :, :]
        prot_basis = qcqp._prot_basis

        done: list[np.ndarray] = []
        for col in new_cols:
            for j in range(prot_basis.shape[1]):
                col -= crdot(prot_basis[:, j], col) * prot_basis[:, j]
            for j in range(npr, qcqp.k):
                col -= crdot(qcqp.Pdiags[:, j], col) * qcqp.Pdiags[:, j]
            for prev in done:
                col -= crdot(prev, col) * prev
            col /= la.norm(col)
            done.append(col)
        new_cols = done

    qcqp.Pdiags = np.column_stack([qcqp.Pdiags] + new_cols)
    qcqp.Fs = np.column_stack(
        [qcqp.Fs]
        + [qcqp.prob.A2H @ (c.conj() * qcqp.prob.s1) for c in new_cols]
    )
    qcqp.current_grad = None


def merge_lead_constraints(qcqp: "_MatrixFreeQCQPImpl", merged_num: int) -> None:
    """Merge the first merged_num unprotected constraints into one.

    The merged column is the multiplier-weighted sum, normalized; its
    multiplier is the normalization factor, so sum(lags_j P_j) — and hence
    the dual value — is unchanged.
    """
    s = qcqp.n_protected
    e = s + merged_num - 1
    if merged_num < 2 or e >= qcqp.k:
        raise ValueError("insufficient unprotected constraints to merge")
    lags = qcqp.current_lags
    assert lags is not None
    d_merged = qcqp.Pdiags[:, s : e + 1] @ lags[s : e + 1]
    Pnorm = la.norm(d_merged)
    d_merged = d_merged / Pnorm

    keep = np.ones(qcqp.k, dtype=bool)
    keep[s:e] = False  # column e becomes the merged slot (lands at index s)
    Pdiags = qcqp.Pdiags[:, keep]
    Pdiags[:, s] = d_merged
    Fs = qcqp.Fs[:, keep]
    Fs[:, s] = qcqp.prob.A2H @ (d_merged.conj() * qcqp.prob.s1)
    qcqp.Pdiags, qcqp.Fs = Pdiags, Fs

    qcqp.current_lags = lags[keep]
    qcqp.current_lags[s] = Pnorm
    qcqp.current_grad = None


def orthonormalize_constraints(qcqp: "_MatrixFreeQCQPImpl") -> None:
    """Orthonormalize unprotected columns against the protected span.

    Multipliers are compensated so sum(lags_j P_j) is unchanged
    (replicates the dolphindes lag transform with solve_triangular).
    """
    npr = qcqp.n_protected
    if qcqp.k <= npr:
        return
    realext = np.vstack((np.real(qcqp.Pdiags), np.imag(qcqp.Pdiags)))
    prot_q, prot_r = la.qr(realext[:, :npr], mode="economic")
    B = prot_q.T @ realext[:, npr:]
    free_q, free_r = la.qr(realext[:, npr:] - prot_q @ B, mode="economic")

    qcqp.Pdiags[:, npr:] = free_q[: qcqp.n, :] + 1j * free_q[qcqp.n :, :]
    lags = qcqp.current_lags
    assert lags is not None
    lags[:npr] += la.solve_triangular(prot_r, B) @ lags[npr:]
    lags[npr:] = free_r @ lags[npr:]
    qcqp.Fs = qcqp.prob.fs_columns(qcqp.Pdiags)


# ----------------------------------------------------------------------
# GCD pipeline
# ----------------------------------------------------------------------


def gcd_round(qcqp: "_MatrixFreeQCQPImpl", params: MatrixFreeGCDParams) -> None:
    """One fixed-weight GCD run (replicates dolphindes run_gcd)."""
    opt_params = params.opt_params or MatrixFreeBFGSParams(verbose=qcqp.verbose - 1)

    qcqp.current_lags = qcqp.find_feasible_lags()
    if params.orthonormalize:
        orthonormalize_constraints(qcqp)

    gcd_iter = 0
    gcd_prev_dual = np.inf
    nonfinite_streak = 0
    stall_ref: Optional[float] = None
    confirm_used = 0
    rescue_depths: list[int] = []
    qcqp.last_gcd_rescue_depths = rescue_depths

    while True:
        gcd_iter += 1
        qcqp.bfgs(init_lags=qcqp.current_lags, params=opt_params)
        assert qcqp.current_dual is not None
        if qcqp.verbose > 0:
            print(f"GCD iteration #{gcd_iter}: dual = {qcqp.current_dual}")

        # smallest eigenpair, computed once and shared by add_source (the
        # persistent fake-source schedule) and the min_aeig constraint-
        # generation direction below -- both want the same eigenvector of
        # A(current_lags), so solving it twice would be a wasted eigensolve
        v_min: Optional[np.ndarray] = None
        try:
            v_min, _ = qcqp._psd_penalty(qcqp.current_lags)
        except (IndefiniteError, ValueError):
            pass
        if v_min is not None:
            qcqp.add_source(v_min)

        # termination checks — order replicates dolphindes:
        # nonfinite cap -> max-iter -> stall confirmation -> periodic stall
        if not np.isfinite(qcqp.current_dual):
            nonfinite_streak += 1
            if nonfinite_streak >= 2:
                if qcqp.verbose > 0:
                    print("Stopping GCD: dual non-finite twice in a row.")
                break
        else:
            nonfinite_streak = 0
        if gcd_iter > params.max_gcd_iter_num:
            break
        if stall_ref is not None:
            confirm_used += 1
            if np.isfinite(qcqp.current_dual) and (
                stall_ref - qcqp.current_dual > params.gcd_tol * abs(stall_ref)
            ):
                rescue_depths.append(confirm_used)
                gcd_prev_dual = qcqp.current_dual
                stall_ref = None
                confirm_used = 0
            elif confirm_used >= params.stall_confirm_iters:
                break
        elif gcd_iter % params.gcd_iter_period == 0 and np.isfinite(
            qcqp.current_dual
        ):
            if (
                gcd_prev_dual - qcqp.current_dual
                < params.gcd_tol * abs(gcd_prev_dual)
            ):
                if params.stall_confirm_iters > 0:
                    stall_ref = min(gcd_prev_dual, qcqp.current_dual)
                    confirm_used = 0
                else:
                    break
            else:
                gcd_prev_dual = qcqp.current_dual

        # constraint generation
        pr = qcqp.prob
        new_cols: list[np.ndarray] = []
        if qcqp.current_xstar is not None:
            xs = qcqp.current_xstar
            max_viol = (2 * pr.s1 - pr.A1H @ xs) * (pr.A2 @ xs).conj()
            if la.norm(max_viol) >= 1e-14:
                new_cols.append(max_viol)

        if v_min is not None:
            min_aeig = (pr.A1H @ v_min) * (pr.A2 @ v_min).conj()
            # replicates a dolphindes quirk: this divides ELEMENTWISE by each
            # entry's modulus (a phase map), not by the vector 2-norm; the
            # generated constraint direction depends on it
            min_aeig = min_aeig / np.sqrt(np.real(min_aeig.conj() * min_aeig))
            if np.all(np.isfinite(min_aeig)):
                new_cols.append(min_aeig)

        add_constraints(qcqp, new_cols, orthonormalize=params.orthonormalize)
        if qcqp.k > params.max_cstrt_num:
            merge_lead_constraints(qcqp, qcqp.k - params.max_cstrt_num + 1)


def run_gcd(
    qcqp: "_MatrixFreeQCQPImpl", params: Optional[MatrixFreeGCDParams] = None
) -> dict:
    """Full annealed pipeline: GCD rounds while annealing the barrier.

    Only rounds ending inside the barrier domain (finite barrier value, hence
    PSD-verified multipliers) count as bounds; the best such round's recorded
    values are reported.
    """
    params = params or MatrixFreeGCDParams()
    assert qcqp.barrier_weight > 0
    assert qcqp.n_protected > qcqp.barrier_idx
    assert params.max_cstrt_num >= qcqp.n_protected + 2
    # the outer annealing loop below has no hard iteration cap (unlike
    # gcd_round's inner loop, which always terminates by max_gcd_iter_num) --
    # it relies entirely on barrier_weight shrinking past final_weight, so a
    # misconfigured anneal_factor outside (0, 1) would spin forever
    assert 0 < params.anneal_factor < 1, "anneal_factor must be in (0, 1)"

    history = []
    best_raw = np.inf
    best_g1 = np.nan
    round_i = 0
    while True:
        round_i += 1
        gcd_round(qcqp, params)
        # current_xstar is already A(current_lags)'s solution -- bfgs() set
        # both together in the same statement -- so raw_value/barrier_value/g1
        # (== gq) are pure dot products away, no re-solve needed.
        # NOTE: not bit-identical to a naive two-extra-CG-solve approach --
        # those solves would also populate the shared CG warm-start cache,
        # so avoiding them shifts later solves' convergence paths at the
        # CG-rtol level (empirically, in the source project: bound moves
        # ~0.15%, well within established scatter, and solve count *drops*
        # -- a net win, not a regression).
        if qcqp.current_xstar is not None:
            assert qcqp.current_lags is not None
            raw_value, barrier_value, g1 = qcqp._raw_and_barrier_from_xstar(
                qcqp.current_lags, qcqp.current_xstar
            )
        else:
            raw_value, barrier_value, g1 = np.inf, np.inf, np.nan
        history.append((qcqp.barrier_weight, raw_value, g1))
        if qcqp.verbose > 0:
            print(
                f"round {round_i}: w = {qcqp.barrier_weight:.2e}, "
                f"raw dual = {raw_value:.10f}, "
                f"barrier = {barrier_value:+.2e}, g_1 = {g1:+.4e}"
            )
        if np.isfinite(barrier_value) and raw_value < best_raw:
            best_raw = raw_value
            best_g1 = g1
        if qcqp.barrier_weight <= params.final_weight:
            break
        qcqp.barrier_weight *= params.anneal_factor
        qcqp.decay_sources()

    return {
        "bound": best_raw,
        "g1": best_g1,
        "rounds": round_i,
        "history": history,
        "rescue_depths": list(qcqp.last_gcd_rescue_depths),
        "solve_stats": dict(qcqp.solve_stats),
        # not part of solve_stats (that dict is only CGSolver's own counters):
        # counts _line_search phase-2 loops cut off by _MAX_IMPROVE_SHRINKS,
        # the exact noise-driven-stall pattern behind a documented past
        # incident (see qcqp.py); should be 0 in any healthy run, and was
        # previously invisible outside of reading qcqp.line_search_stalls
        # directly
        "line_search_stalls": qcqp.line_search_stalls,
    }
