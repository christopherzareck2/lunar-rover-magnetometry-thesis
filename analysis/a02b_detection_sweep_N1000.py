"""a02b - Detection sweep (Figure 4) at N = 1000 per point, consistent with the baseline.

Every point uses N = 1000, and the 1x point uses the baseline seed of a02, so it reproduces the
baseline exactly. Geometry, noise and solver are imported from a02.

Run:  python analysis/a02b_detection_sweep_N1000.py
Out:  results/analysis/a02b_detection_sweep_N1000.json
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import a02_isolated_outcrop_paper_geometry as A02          # sets engine path and cwd
from inversion_helpers import prism_cell_operator, seed_for

N = 1000


def main():
    t0 = time.time()
    out = {"meta": dict(script="analysis/a02b_detection_sweep_N1000.py", N=N, ratios=A02.RATIOS,
                        seeds="ratio 1 uses seed_for('a02:base:<survey>') (identical to the a02 baseline); "
                              "others seed_for('a02b:sweep:<survey>:<ratio>')"), "detection_sweep": {}}
    for name in A02.SURVEYS:
        G = prism_cell_operator(A02.BND, A02.survey(name))
        rows = []
        for r in A02.RATIOS:
            sd = seed_for(f"a02:base:{name}") if r == 1 else seed_for(f"a02b:sweep:{name}:{r}")
            s = A02.recover(G, A02.M_TRUE * r, N, sd); s["ratio"] = r
            rows.append(s)
            print(f"[{name}] ratio {r:6.1f}: theta med {s['dir_err_deg']['median']:6.2f} mean {s['dir_err_deg']['mean']:6.2f} "
                  f"|dM| med {s['dM_abs_pct']['median']:8.3f}%", flush=True)
        out["detection_sweep"][name] = rows
    out["meta"]["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a02b_detection_sweep_N1000.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
