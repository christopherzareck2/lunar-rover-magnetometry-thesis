"""Check the closed-form prism kernel (ml/common.py) against the SimPEG integral operator the thesis uses
(engine/inversion_helpers.prism_cell_operator), for the exposed prism, buried prisms, root slabs and rigidly
mis-posed prisms on the two-loop survey, and time both.

Run:  python ml/validate_forward.py
Out:  ml/results/forward_validation.json
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from isolated_outcrop_1cm import per_receiver_kernel
from inversion_helpers import prism_cell_operator


def main():
    sx = C.A9.two_loops(dx=0.5)
    cases = {}
    rng = np.random.default_rng(C.seed_for("ml:validate"))
    for name, bnd in {"exposed": C.BND, "buried_0.37m": C.buried_bounds(0.37), "buried_1m": C.buried_bounds(1.0),
                      "slab_0.6-0.7m": (C.BND[0], C.BND[1], C.BND[2], C.BND[3], C.BND[4] - 0.7, C.BND[4] - 0.6)}.items():
        t0 = time.time(); Gs = prism_cell_operator(bnd, sx) * C.MREF; ts = time.time() - t0
        t0 = time.time(); Gc = C.operator(bnd, sx); tc = time.time() - t0
        rel = float(np.abs(Gc - Gs).max() / np.abs(Gs).max())
        cases[name] = {"max_abs_err_over_max": rel, "t_simpeg_s": ts, "t_closed_form_s": tc}
        print(f"{name:14s} rel err {rel:.2e}  SimPEG {ts:.3f} s  closed form {tc:.4f} s", flush=True)
    # rigid mis-pose: thesis route (per-receiver SimPEG kernel in the body frame, then rotated) vs closed form
    for j in range(3):
        az, el = rng.uniform(-3, 3, 2); T = rng.uniform(-0.15, 0.15, 3)
        R = C.rot_az_el(az, el)
        Q = (sx - T) @ R
        Aq = per_receiver_kernel(C.BND, Q) * C.MREF
        Gs = np.einsum("ij,rjk,lk->ril", R, Aq, R).reshape(-1, 3)
        Gc = C.posed_operator(C.BND, sx, R, T)
        rel = float(np.abs(Gc - Gs).max() / np.abs(Gs).max())
        cases[f"posed_{j}"] = {"az_deg": float(az), "el_deg": float(el), "T_m": T.tolist(), "max_abs_err_over_max": rel}
        print(f"posed_{j}: rel err {rel:.2e}", flush=True)
    worst = max(v["max_abs_err_over_max"] for v in cases.values())
    out = {"kernel_sign": C.KERNEL_SIGN, "n_stations": int(len(sx)), "cases": cases, "worst_rel_err": worst,
           "pass": bool(worst < 1e-8)}
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "forward_validation.json"), "w"), indent=1)
    print("PASS" if out["pass"] else "FAIL", f"worst {worst:.2e}")


if __name__ == "__main__":
    main()
