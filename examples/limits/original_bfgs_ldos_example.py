"""
LDOS dual bound via dolphindes's original Cholesky-based GCD pipeline, BFGS.

Sister script to examples/limits/matrix_free_ldos_example.py: builds the
IDENTICAL LDOS problem (same physics, same dl/PML/design-region setup) but
with Photonics_TM_FDFD(matrix_free=False) (the default), so the dual problem
is solved by the original SparseSharedProjQCQP -- CHOLMOD factorizations via
scikit-sparse, no CG, no log barrier -- refined by the same
dolphindes.cvxopt.gcd.run_gcd/GCDHyperparameters pipeline the matrix-free
script uses (add/merge shared-projection constraints, tighten the bound over
GCD iterations). Upstream's run_gcd hardcodes its internal per-iteration
solve to "newton" (dolphindes/cvxopt/gcd.py, not configurable via
GCDHyperparameters); to get the requested BFGS variant without modifying
dolphindes itself, this script temporarily wraps
QCQP.solve_current_dual_problem so every call GCD makes internally uses
"bfgs" instead, for the duration of the run_gcd call only.

Run both scripts on the same --px-per-length to compare directly.

Usage:
    python examples/limits/original_bfgs_ldos_example.py [--px-per-length 40]
        [--chi (4+1e-4j)] [--max-cstrt-num 10] [--max-gcd-iter-num 50]
        [--gcd-iter-period 5] [--gcd-tol 1e-2] [--seed 0] [--verbose 1]
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp

from dolphindes import geometry, photonics
from dolphindes.cvxopt import GCDHyperparameters, SparseSharedProjQCQP


def build_ldos_problem(px_per_length: int, chi: complex = 4 + 1e-4j, verbose: int = 0):
    """Build the TM LDOS maximization problem (mirrors LDOS_gcd.ipynb exactly).

    A point dipole source sits a distance 0.1 from a 0.5 x 0.5 design region
    of material chi, surrounded by PML.
    """
    wavelength = 1.0
    omega = 2 * np.pi / wavelength
    dl = 1 / px_per_length
    Npmlsep = int(0.5 / dl)
    Npmlx, Npmly = int(0.5 / dl), int(0.5 / dl)
    Mx, My = int(0.5 / dl), int(0.5 / dl)
    Dx = int(0.1 / dl)
    Nx = int(Npmlx * 2 + Npmlsep * 2 + Dx + Mx)
    Ny = int(Npmly * 2 + Npmlsep * 2 + My)

    cx, cy = Npmlx + Npmlsep, Ny // 2

    ji = np.zeros((Nx, Ny), dtype=complex)
    ji[cx, cy] = 1.0 / dl / dl
    design_mask = np.zeros((Nx, Ny), dtype=bool)
    design_mask[
        Npmlx + Npmlsep + Dx : Npmlx + Npmlsep + Dx + Mx,
        Npmly + Npmlsep : Npmly + Npmlsep + My,
    ] = True
    ndof = int(np.sum(design_mask))
    chi_background = np.zeros((Nx, Ny), dtype=complex)

    fdfd_geometry = geometry.CartesianFDFDGeometry(
        Nx=Nx, Ny=Ny, Npmlx=Npmlx, Npmly=Npmly, dx=dl, dy=dl
    )
    ldos_problem = photonics.Photonics_TM_FDFD(
        omega=omega,
        geometry=fdfd_geometry,
        chi=chi,
        des_mask=design_mask,
        ji=ji,
        chi_background=chi_background,
        sparseQCQP=True,
        matrix_free=False,
    )

    ei = ldos_problem.get_ei(ji, update=True)
    vac_ldos = -np.sum(1 / 2 * np.real(ji.conj() * ei) * dl * dl)

    ei_design = ei[ldos_problem.des_mask]
    s0_p = -(1 / 4) * 1j * omega * ei_design.conj()
    A0_p = sp.csc_array(np.zeros((ndof, ndof), dtype=complex))
    ldos_problem.set_objective(s0=s0_p, A0=A0_p, c0=vac_ldos, denseToSparse=True)

    ldos_problem.setup_QCQP(Pdiags="global", verbose=verbose)

    return ldos_problem, float(vac_ldos), ndof


def run_gcd_with_bfgs(
    qcqp: SparseSharedProjQCQP, gcd_params: GCDHyperparameters
) -> None:
    """Run dolphindes's stock GCD pipeline, forcing its internal solve to BFGS.

    run_gcd (dolphindes/cvxopt/gcd.py) always calls
    solve_current_dual_problem("newton", ...) internally; there is no
    GCDHyperparameters field to change that. This wraps the method for the
    duration of the call so every internal solve uses "bfgs" instead, then
    restores the original method -- the QCQP instance is left otherwise
    unmodified.
    """
    original = qcqp.solve_current_dual_problem

    def forced_bfgs(method, opt_params=None, init_lags=None):
        return original("bfgs", opt_params=opt_params, init_lags=init_lags)

    qcqp.solve_current_dual_problem = forced_bfgs  # type: ignore[method-assign]
    try:
        qcqp.run_gcd(gcd_params=gcd_params)
    finally:
        qcqp.solve_current_dual_problem = original  # type: ignore[method-assign]


def main() -> None:
    """Parse CLI args, build the LDOS problem, run GCD with BFGS, report results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px-per-length", type=int, default=40)
    parser.add_argument("--chi", type=complex, default=4 + 1e-4j)
    parser.add_argument("--max-cstrt-num", type=int, default=10)
    parser.add_argument("--max-gcd-iter-num", type=int, default=50)
    parser.add_argument("--gcd-iter-period", type=int, default=5)
    parser.add_argument("--gcd-tol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)

    t = time.time()
    ldos_problem, vac_ldos, ndof = build_ldos_problem(
        args.px_per_length, chi=args.chi, verbose=args.verbose
    )
    setup_time = time.time() - t
    print(f"Vacuum LDOS: {vac_ldos}")
    print(f"Design dofs: {ndof}, setup took {setup_time:.2f}s")

    assert isinstance(ldos_problem.QCQP, SparseSharedProjQCQP)

    gcd_params = GCDHyperparameters(
        max_proj_cstrt_num=args.max_cstrt_num,
        orthonormalize=True,
        opt_params=None,
        max_gcd_iter_num=args.max_gcd_iter_num,
        gcd_iter_period=args.gcd_iter_period,
        gcd_tol=args.gcd_tol,
    )

    t = time.time()
    run_gcd_with_bfgs(ldos_problem.QCQP, gcd_params)
    solve_time = time.time() - t

    dual = ldos_problem.QCQP.current_dual
    print("\n=== dolphindes original (Cholesky, GCD, BFGS) LDOS results ===")
    print(f"resolution (px/wavelength):  {args.px_per_length}")
    print(f"design dofs:                 {ndof}")
    print(f"vacuum LDOS:                 {vac_ldos}")
    print(f"dual bound:                  {dual}")
    print(f"bound enhancement (dual/vac): {dual / vac_ldos}")
    print(f"final constraint count:      {ldos_problem.QCQP.n_proj_constr}")
    print(f"setup time:                  {setup_time:.3f}s")
    print(f"GCD solve time:              {solve_time:.3f}s")


if __name__ == "__main__":
    main()
