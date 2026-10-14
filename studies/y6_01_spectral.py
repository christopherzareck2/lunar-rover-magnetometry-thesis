"""Y6-01: wavenumber sensitivity of the estimator with the one-sided Fourier multiplicity included.

Released code (analysis/a06_isolated_robustness.py::part_F) sums |rfft(w)|^2 of the nine weight series of K = (G^T G)^-1 G^T
for the single drive-around loop and takes cumulative shares, without counting each bin above DC twice. For a real
series of length n, Parseval's theorem is  sum_j w_j^2 = (1/n) [ |W_0|^2 + 2 sum_{0<k<n/2} |W_k|^2 (+ |W_{n/2}|^2 if n even) ],
so the released shares over-weight the DC bin (review finding Y6-01, see docs/known-issues.md).

What this script does (a modified copy of a06 part_F; nothing in analysis/ is edited):
  * doubles every bin with 0 < k < n/2 (n = 2399 is odd, so there is no Nyquist bin) and checks Parseval exactly;
  * reports the shares for (a) all nine series (the trace of the estimator covariance under white noise, as coded),
    (b) the direction of the recovered vector, P_perp K with P_perp = I - u u^T, which is what the direction error
    responds to in the small-error limit, and (c) the intensity, u^T K, for u = -z, +x, +y and 45 deg inclination;
  * converts bins to along-track wavelength (loop length / k) and to period at 1 cm/s.
For white noise the per-bin split is exact for the finite record (Parseval on an orthogonal basis). Weighting the bins
by a coloured spectrum S(k) assumes the noise is stationary and periodic around the loop (circulant covariance).

Run:  python studies/y6_01_spectral.py      (seconds)
Out:  studies/results/y6_01_spectral.json
"""
from __future__ import annotations

import numpy as np

import _common as C

SPEED = 0.01        # m/s, the thesis's assumed drive speed
NB = 8              # bins reported individually


def shares(series: np.ndarray, n: int, double: bool) -> np.ndarray:
    """Per-bin share of sum |rfft|^2 over the rows of `series` (each a length-n weight series)."""
    P = np.zeros(n // 2 + 1)
    for w in series:
        P += np.abs(np.fft.rfft(w)) ** 2
    if double:
        P[1:(n + 1) // 2] *= 2.0          # every bin except DC (and Nyquist when n is even)
    return P / P.sum(), P


def summarize(sh: np.ndarray, kbin_len: float) -> dict:
    c = np.cumsum(sh)
    k95 = int(np.searchsorted(c, 0.95)); k50 = int(np.searchsorted(c, 0.50))
    lam = lambda k: (kbin_len / k) if k > 0 else float("inf")
    return {"share_by_bin_pct": [float(100 * v) for v in sh[:NB]],
            "cumulative_by_bin_pct": [float(100 * v) for v in c[:NB]],
            "dc_pct": float(100 * sh[0]),
            "share_lambda_ge_12m_pct": float(100 * c[int(np.floor(1.01 * kbin_len / 12.0))]),   # bins with L/k >= 12 m (1% tolerance: the 11.995 m bin counts)
            "share_lambda_ge_6m_pct": float(100 * c[int(np.floor(1.01 * kbin_len / 6.0))]),     # bins with L/k >= 6 m (same tolerance)
            "first_bin_reaching_50pct": k50, "first_bin_reaching_95pct": k95,
            "wavelength_at_95pct_m": lam(k95), "period_at_95pct_min": lam(k95) / SPEED / 60.0}


def main():
    out = {"meta": C.provenance(__file__, {"finding": "Y6-01", "speed_m_per_s": SPEED,
                                           "note": "deterministic; no random draws"})}
    for name in ("drive_around", "drive_by"):
        sx = C.A6.survey(name)
        G = C.prism_cell_operator(C.BND, sx)
        K = C.A6.lsK(G)                                  # 3 x 3n
        n = len(sx)
        W = K.reshape(3, n, 3)                          # [M component, station, field component]
        path_len = float(np.sum(np.linalg.norm(np.diff(sx[:, :2], axis=0), axis=1)))
        kbin_len = n * C.A6.SPACING                    # length represented by one DFT period (n samples x 1 cm)
        nine = np.array([W[i, :, j] for i in range(3) for j in range(3)])
        sh_coded, Pc = shares(nine, n, double=False)
        sh_fixed, Pf = shares(nine, n, double=True)
        parseval = float(Pf.sum() / n / np.sum(nine ** 2))
        res = {"n_stations": n, "path_length_m": path_len, "dft_period_m": kbin_len,
               "bin_wavelengths_m": [None] + [kbin_len / k for k in range(1, NB)],
               "parseval_ratio_corrected": parseval,
               "parseval_ratio_as_coded": float(Pc.sum() / n / np.sum(nine ** 2)),
               "all_nine_as_coded": summarize(sh_coded, kbin_len),
               "all_nine_corrected": summarize(sh_fixed, kbin_len)}
        for dn, u in {"-z": [0, 0, -1.0], "+x": [1.0, 0, 0], "+y": [0, 1.0, 0], "inc45": [0.7071068, 0, -0.7071068]}.items():
            u = np.asarray(u) / np.linalg.norm(u)
            Pp = np.eye(3) - np.outer(u, u)
            Wd = np.einsum("ab,bnj->anj", Pp, W)        # direction weights
            Wi = np.einsum("b,bnj->nj", u, W)            # intensity weights
            dser = np.array([Wd[i, :, j] for i in range(3) for j in range(3)])
            iser = np.array([Wi[:, j] for j in range(3)])
            res[f"direction_{dn}"] = summarize(shares(dser, n, True)[0], kbin_len)
            res[f"intensity_{dn}"] = summarize(shares(iser, n, True)[0], kbin_len)
        # the released OU prediction used k_c = k50; recompute it with the corrected k50
        k = np.fft.rfftfreq(n, d=C.A6.SPACING)
        kc_old, kc_new = float(k[np.searchsorted(np.cumsum(sh_coded), 0.5)]), float(k[np.searchsorted(np.cumsum(sh_fixed), 0.5)])
        white = 2 * C.SIGMA_B ** 2 * C.A6.SPACING
        pred = []
        for L in [0.005, 0.01, 0.02, 0.05, 0.1]:
            S = lambda kc: 4 * C.CN.SIGMA_C ** 2 * L / (1 + (2 * np.pi * kc * L) ** 2) + 2 * C.CN.SIGMA_W ** 2 * C.A6.SPACING
            pred.append({"L_m": L, "asd_ratio_kc_as_coded": float(np.sqrt(S(kc_old) / white)),
                         "asd_ratio_kc_corrected": float(np.sqrt(S(kc_new) / white))})
        res["ou_prediction"] = {"kc_as_coded_cyc_per_m": kc_old, "kc_corrected_cyc_per_m": kc_new, "rows": pred}
        out[name] = res
        a, f = res["all_nine_as_coded"], res["all_nine_corrected"]
        print(f"[{name}] n={n} Parseval {parseval:.12f} | DC {a['dc_pct']:.4f} -> {f['dc_pct']:.4f}% | "
              f"lambda>=12m {a['share_lambda_ge_12m_pct']:.4f} -> {f['share_lambda_ge_12m_pct']:.4f}% | "
              f"k95 bin {a['first_bin_reaching_95pct']} -> {f['first_bin_reaching_95pct']} | "
              f"dir(-z) lambda>=12m {res['direction_-z']['share_lambda_ge_12m_pct']:.2f}%")
    print(C.save("y6_01_spectral.json", out))


if __name__ == "__main__":
    main()
