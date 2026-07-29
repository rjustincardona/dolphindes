"""
LDOS dual bound via dolphindes's native matrix-free (CG-based, log-barrier) solver.

Mirrors the canonical LDOS setup in examples/limits/LDOS_gcd.ipynb (a point
dipole source a distance 0.1 from a 0.5 x 0.5 design region of material chi,
surrounded by PML), but sets Photonics_TM_FDFD(matrix_free=True) so the dual
problem is solved by dolphindes.cvxopt.matrix_free.MatrixFreeSharedProjQCQP
(CG solves, no Cholesky factorization, no dense/sparse matrix ever
assembled/factorized for the dual solve) instead of the Cholesky-based
SparseSharedProjQCQP/DenseSharedProjQCQP. Problem assembly (building
A0/A1/A2/s0/s1 from the physics) is identical either way -- matrix_free only
changes how the dual problem is solved.

Usage:
    python examples/limits/matrix_free_ldos_example.py [--px-per-length 40]
        [--max-cstrt-num 10] [--max-gcd-iter-num 50] [--gcd-iter-period 1]
        [--gcd-tol 1e-2] [--stall-confirm-iters 3] [--anneal-factor 1e-2]
        [--final-weight 1e-8] [--seed 0] [--verbose 1]
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp

from dolphindes import geometry, photonics
from dolphindes.cvxopt.matrix_free import MatrixFreeGCDParams, MatrixFreeSharedProjQCQP


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
        matrix_free=True,
    )

    ei = ldos_problem.get_ei(ji, update=True)
    vac_ldos = -np.sum(1 / 2 * np.real(ji.conj() * ei) * dl * dl)

    ei_design = ei[ldos_problem.des_mask]
    s0_p = -(1 / 4) * 1j * omega * ei_design.conj()
    A0_p = sp.csc_array(np.zeros((ndof, ndof), dtype=complex))
    ldos_problem.set_objective(s0=s0_p, A0=A0_p, c0=vac_ldos, denseToSparse=True)

    ldos_problem.setup_QCQP(Pdiags="global", verbose=verbose)

    return ldos_problem, float(vac_ldos), ndof


def main() -> None:
    """Parse CLI args, build the LDOS problem, run matrix-free GCD, report results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px-per-length", type=int, default=40)
    parser.add_argument("--chi", type=complex, default=4 + 1e-4j)
    parser.add_argument("--max-cstrt-num", type=int, default=10)
    parser.add_argument("--max-gcd-iter-num", type=int, default=50)
    parser.add_argument("--gcd-iter-period", type=int, default=1)
    parser.add_argument("--gcd-tol", type=float, default=1e-2)
    parser.add_argument("--stall-confirm-iters", type=int, default=3)
    parser.add_argument("--anneal-factor", type=float, default=1e-2)
    parser.add_argument("--final-weight", type=float, default=1e-8)
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

    assert isinstance(ldos_problem.QCQP, MatrixFreeSharedProjQCQP)

    params = MatrixFreeGCDParams(
        max_cstrt_num=args.max_cstrt_num,
        max_gcd_iter_num=args.max_gcd_iter_num,
        gcd_iter_period=args.gcd_iter_period,
        gcd_tol=args.gcd_tol,
        stall_confirm_iters=args.stall_confirm_iters,
        anneal_factor=args.anneal_factor,
        final_weight=args.final_weight,
    )

    t = time.time()
    out = ldos_problem.QCQP.run_gcd(params)
    solve_time = time.time() - t

    stats = out["solve_stats"]
    avg = stats["cg_iters"] / max(stats["cg_solves"], 1)
    print("\n=== dolphindes matrix-free LDOS log-barrier results ===")
    print(f"resolution (px/wavelength):  {args.px_per_length}")
    print(f"design dofs:                 {ndof}")
    print(f"vacuum LDOS:                 {vac_ldos}")
    print(f"dual bound (raw, no barrier): {out['bound']}")
    print(f"bound enhancement (dual/vac): {out['bound'] / vac_ldos}")
    print(f"final g_1:                   {out['g1']:+.4e}")
    print(f"annealing rounds:            {out['rounds']}")
    print(f"setup time:                  {setup_time:.3f}s")
    print(f"total solve time:            {solve_time:.3f}s")
    print(
        f"CG solves:                   {stats['cg_solves']} "
        f"({stats['cg_iters']} total iters, {avg:.0f} avg/solve, "
        f"{stats['cg_indefinite']} indefinite, {stats['cg_noconv']} nonconv, "
        f"{stats['cg_warm_hits']} warm starts)"
    )


if __name__ == "__main__":
    main()
