#!/usr/bin/env python3
"""Fisher information check for thesis Section 7.2: Schur complement and the effect of a second loop.

Usage: python studies/fisher/fisher_check.py [REPO_ROOT]   (default: this checkout)
It imports ml/common.py (closed-form prism operator, validated against SimPEG to 7e-14) and the a06/a09 survey
helpers, read-only, and writes fisher_check.json next to this script.

Checks:
 1. Schur complement: [(A^T A)^{-1}]_mm == (G^T P_N^perp G)^{-1}  (N = offsets, or offsets + gradient)
 2. Scatter identity: G^T P_E^perp G == sum_i (G_i - Gbar)^T (G_i - Gbar)
 3. Two-set (parallel-axis) identity: S = S_1 + S_2 + P1 P2/(P1+P2) (Gbar_1 - Gbar_2)^T (Gbar_1 - Gbar_2)
 4. Information lost to the offsets on the 1 m loop (P1 Gbar_1^T Gbar_1) and how much a 10 m leg or a
    3 m loop restores, component by component.
 5. Gradient terms: extra information lost when the 5 gradient columns are fitted, centred vs 0.5 m off.
"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(root, "ml"))
cwd = os.getcwd()
import common as C   # noqa: E402  (imports a06/a09, which chdir)
os.chdir(cwd)
A6, A9 = C.A6, C.A9
SIG = C.SIGMA_B

loop1 = A6.survey("drive_around")
loop3 = A6.survey("drive_around", standoff=3.0)
z0 = loop1[0, 2]
leg = np.c_[np.linspace(-13.0, -3.0 - 0.01, 1000), np.full(1000, -3.0), np.full(1000, z0)]

def blocks(sx):
    G = C.operator(C.BND, sx)                  # (3P, 3), nT per M_ref
    return G, G.reshape(-1, 3, 3)              # station blocks G_i (3x3)

def E_cols(n):
    return np.tile(np.eye(3), (n, 1))

def info(G, N=None):
    if N is None or N.shape[1] == 0:
        return G.T @ G / SIG**2
    Qn, _ = np.linalg.qr(N)
    PG = G - Qn @ (Qn.T @ G)
    return G.T @ PG / SIG**2

def crlb_block(G, N):
    A = np.column_stack([G, N]) if N is not None else G
    return SIG**2 * np.linalg.inv(A.T @ A)[:3, :3]

out = {"sigma_B_nT": SIG, "units": "information in (M_ref)^-2; G in nT per M_ref"}
designs = {"loop_1m": loop1, "loop_1m+10m_leg": np.vstack([leg, loop1]), "loop_1m+loop_3m": np.vstack([loop1, loop3])}
res = {}
for name, sx in designs.items():
    G, Gi = blocks(sx); n = len(sx)
    E = E_cols(n)
    Ng = A6.gradient_basis(sx)[0]
    I0 = info(G); IE = info(G, E); IEg = info(G, np.column_stack([E, Ng]))
    # 1. Schur complement vs block of inverse
    s1 = float(np.abs(np.linalg.inv(IE) - crlb_block(G, E)).max() / np.abs(crlb_block(G, E)).max())
    s1g = float(np.abs(np.linalg.inv(IEg) - crlb_block(G, np.column_stack([E, Ng]))).max()
                / np.abs(crlb_block(G, np.column_stack([E, Ng]))).max())
    # 2. scatter identity
    Gbar = Gi.mean(0)
    S = np.einsum("nij,nik->jk", Gi - Gbar, Gi - Gbar) / SIG**2
    s2 = float(np.abs(S - IE).max() / np.abs(IE).max())
    res[name] = {"P": n, "schur_rel_err_offsets": s1, "schur_rel_err_offsets_gradient": s1g,
                 "scatter_identity_rel_err": s2,
                 "info_diag_no_offsets": np.diag(I0).tolist(), "info_diag_offsets": np.diag(IE).tolist(),
                 "info_diag_offsets_gradient": np.diag(IEg).tolist(),
                 "crlb_sd_offsets_Mref": np.sqrt(np.diag(np.linalg.inv(IE))).tolist(),
                 "crlb_sd_no_offsets_Mref": np.sqrt(np.diag(np.linalg.inv(I0))).tolist(),
                 "Gbar": Gbar.tolist()}

# 3-4. two-set decomposition for the second loop and the leg
G1, G1i = blocks(loop1); P1 = len(loop1); Gb1 = G1i.mean(0)
lost1 = P1 * Gb1.T @ Gb1 / SIG**2           # information the offsets remove on the 1 m loop alone
for name, extra in (("loop_3m", loop3), ("10m_leg", leg)):
    G2, G2i = blocks(extra); P2 = len(extra); Gb2 = G2i.mean(0)
    S1 = np.einsum("nij,nik->jk", G1i - Gb1, G1i - Gb1) / SIG**2
    S2 = np.einsum("nij,nik->jk", G2i - Gb2, G2i - Gb2) / SIG**2
    Bt = P1 * P2 / (P1 + P2) * (Gb1 - Gb2).T @ (Gb1 - Gb2) / SIG**2
    sx = np.vstack([loop1, extra]); G, _ = blocks(sx)
    IE = info(G, E_cols(len(sx)))
    rel = float(np.abs(S1 + S2 + Bt - IE).max() / np.abs(IE).max())
    res["decomp_" + name] = {
        "P1": P1, "P2": P2, "identity_rel_err": rel,
        "frac_P2_over_P": P2 / (P1 + P2),
        "Gbar2_over_Gbar1_frobenius": float(np.linalg.norm(Gb2) / np.linalg.norm(Gb1)),
        "lost_on_loop1_diag": np.diag(lost1).tolist(),
        "between_term_diag": np.diag(Bt).tolist(),
        "within_S2_diag": np.diag(S2).tolist(),
        "S1_diag": np.diag(S1).tolist(),
        "restored_fraction_of_lost_diag": (np.diag(Bt) / np.diag(lost1)).tolist(),
        "gain_over_loop1_offsets_diag": ((np.diag(S2) + np.diag(Bt))).tolist(),
    }

# 5. gradient cost, centred vs off centre (two loops)
grad = {}
for dx in (0.0, 0.5):
    sx = A9.two_loops(dx=dx)
    G, _ = blocks(sx); E = E_cols(len(sx)); Ng = A6.gradient_basis(sx)[0]
    IE = info(G, E); IEg = info(G, np.column_stack([E, Ng]))
    grad[f"dx_{dx}"] = {"info_diag_offsets": np.diag(IE).tolist(), "info_diag_offsets_gradient": np.diag(IEg).tolist(),
                        "trace_crlb_ratio_grad_over_offsets": float(np.trace(np.linalg.inv(IEg)) / np.trace(np.linalg.inv(IE)))}
res["gradient_cost_two_loops"] = grad
out["results"] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
