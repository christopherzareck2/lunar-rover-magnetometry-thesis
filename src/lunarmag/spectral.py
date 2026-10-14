"""Wavenumber band of the estimator weights (thesis Section 5.1 and Appendix Section B.3, the "twenty minutes" argument).

For stationary noise with one-sided spatial power spectral density S(k), the variance of a linear estimate
p_hat_i = sum_t K_i(t) n(t) is approximately sum_k S(k) |K_hat_i(k)|^2 (a periodic, finite-record approximation).
The band where |K_hat(k)|^2 holds its weight is therefore the band where noise matters.

``estimator_power`` returns that weight per real-FFT bin. Two normalizations are offered:

* ``one_sided=True`` (default, correct): ``numpy.fft.rfft`` returns only non-negative frequencies, so every bin
  except DC (and Nyquist for an even length) stands for two two-sided bins and is doubled. This is the energy
  of the real series by Parseval.
* ``one_sided=False`` reproduces part F of ``analysis/a06_isolated_robustness.py`` (stored results), which sums
  |rfft|^2 without the doubling and so over-weights DC (thesis review finding Y6-01). It is kept so the stored
  numbers can be regenerated; use the corrected form for new statements.

``component="trace"`` sums all nine (output component, data axis) weight series, i.e. the trace of the recovered
vector covariance; ``"direction"`` projects the recovered vector onto the plane normal to the true magnetization
and ``"intensity"`` onto its direction (Y6-01 recommends labeling which one a statement uses).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def estimator_power(K: np.ndarray, n_stations: int, spacing: float, one_sided: bool = True,
                    component: str = "trace", M_true: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Weight of the estimator in each real-FFT wavenumber bin.

    ``K`` is the (3 x 3P) magnetization rows of the estimator (station-major data). Returns (k in cycles per
    metre, power per bin), with power normalized to sum to 1.
    """
    K = np.asarray(K, float)[:3]
    W = K.reshape(3, n_stations, 3)                     # (output component, station, data axis)
    if component == "trace":
        rows = [W[i] for i in range(3)]
    else:
        if M_true is None:
            raise ValueError("component='direction' or 'intensity' needs M_true")
        u = np.asarray(M_true, float) / np.linalg.norm(M_true)
        if component == "intensity":
            rows = [np.einsum("i,ins->ns", u, W)]
        elif component == "direction":
            P = np.eye(3) - np.outer(u, u)
            rows = [np.einsum("i,ins->ns", P[j], W) for j in range(3)]
        else:
            raise ValueError(component)
    P_k = np.zeros(n_stations // 2 + 1)
    for r in rows:
        for j in range(3):
            P_k += np.abs(np.fft.rfft(r[:, j])) ** 2
    if one_sided:
        mult = np.full(P_k.size, 2.0)
        mult[0] = 1.0
        if n_stations % 2 == 0:
            mult[-1] = 1.0                              # Nyquist bin is not duplicated
        P_k = P_k * mult
    k = np.fft.rfftfreq(n_stations, d=spacing)
    return k, P_k / P_k.sum()


def band_quantiles(k: np.ndarray, P: np.ndarray, qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    """Wavenumbers (cycles/m) below which the cumulative weight first reaches each quantile (as stored in a06 F)."""
    c = np.cumsum(P) / P.sum()
    c[-1] = 1.0                                   # exact endpoint, so q = 1 is found despite rounding
    if any(not 0.0 <= q <= 1.0 for q in qs):
        raise ValueError("quantiles must lie in [0, 1]")
    return {f"k{int(round(q * 100))}": float(k[np.searchsorted(c, q)]) for q in qs}


def dc_fraction(P: np.ndarray) -> float:
    """Share of the weight in the zero-wavenumber (survey-mean) bin."""
    return float(P[0] / P.sum())
