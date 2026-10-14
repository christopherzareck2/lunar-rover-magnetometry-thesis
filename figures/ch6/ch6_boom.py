"""Figure 6.5: trade study 2, boom length versus the rover's stray field and the drift it can cause.

Model (stated assumptions, ch6_data.ASSUMPTIONS): the part of the rover field that is neither calibrated nor
fitted is an equivalent residual point dipole whose field at the sensor has a per-axis rms (rms of the three
components, |B|/sqrt(3)) of 0.40 nT, Table 3.1, at r0 = 1 m from the rover's magnetic centroid. The per-axis rms
falls as (r0/r)^3. If the residual moment varies about its survey mean with fractional rms eps (heaters, thermal
changes in currents and soft materials), the sensor sees a drift with pooled per-axis rms eps*B(r); the whole
0.01 nT drift requirement allows eps = 0.01 nT / B(r), and the 0.005 nT rover share of Table 6.4 half that.
A large calibrated moment can vary independently of this residual; the model does not cover that case.
(A) residual rover field versus sensor distance, with the instrument-noise (0.05 nT, Table 3.1) and drift
(0.01 nT, Sec. 9.2) levels for reference. (B) eps_max versus sensor distance.
"""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

import ch6_data as D
from ch6_common import finish, st

FS = st.BASE_PT - 1


def main():
    r = np.geomspace(0.3, 3.0, 200)
    b = D.residual_field(r)
    eps = D.DRIFT_REQ_NT / b * 100.0
    with st.style():
        fig = st.figure("full", 2.5)
        ax = fig.add_axes([0.1, 0.17, 0.37, 0.75])
        bx = fig.add_axes([0.6, 0.17, 0.38, 0.75])
        ax.loglog(r, b, color=st.DIRECTION, lw=1.4, zorder=3)
        for val, lab, ls in ((D.B_RES_R0_NT, "0.40 nT allocation", (0, ())),
                             (0.05, "0.05 nT instrument noise, 1 s", (0, (4, 2.5))),
                             (D.DRIFT_REQ_NT, "0.01 nT drift requirement", (0, (1, 1.5)))):
            ax.axhline(val, color=st.MUTED, lw=0.8, ls=ls, zorder=2)
            ax.text(0.32, val * 1.12, lab, ha="left", va="bottom", fontsize=FS - 0.5, color=st.INK)
        ax.plot([D.R0_M], [D.B_RES_R0_NT], "o", ms=5, mfc="white", mec=st.INK, mew=0.9, zorder=4)
        ax.text(D.R0_M * 0.92, D.B_RES_R0_NT * 0.75, "baseline", ha="right", va="top", fontsize=FS - 0.5)
        ax.set_xlim(0.3, 3.0); ax.set_ylim(0.004, 30)
        ax.set_xlabel("Sensor distance $r$ (m)")
        ax.set_ylabel("Residual rover field, per-axis rms (nT)")
        ax.set_xticks([0.3, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"])

        eps_share = D.DRIFT_ROVER_SHARE_NT / b * 100.0
        bx.loglog(r, eps, color=st.INTENSITY, lw=1.4, zorder=3, label="whole 0.01 nT drift budget")
        bx.loglog(r, eps_share, color=st.INTENSITY, lw=1.2, ls=(0, (4, 2.5)), zorder=3,
                  label="0.005 nT rover share")
        for rr in (0.5, 1.0, 2.0):
            for req, va, dy, dx, ha in ((D.DRIFT_REQ_NT, "bottom", 1.2, 0.93, "right"),
                                        (D.DRIFT_ROVER_SHARE_NT, "top", 0.8, 1.07, "left")):
                e = req / D.residual_field(rr) * 100.0
                bx.plot([rr], [e], "o", ms=3.5, mfc="white", mec=st.INK, mew=0.8, zorder=4)
                bx.text(rr * dx, e * dy, f"{e:.3g}%", ha=ha, va=va, fontsize=FS - 1)
        bx.legend(loc="upper left", frameon=False, handlelength=1.8)
        bx.set_xlim(0.3, 3.0); bx.set_ylim(0.05, 1000)
        bx.set_xticks([0.3, 0.5, 1, 2, 3]); bx.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
        bx.set_xlabel("Sensor distance $r$ (m)")
        bx.set_ylabel("Allowed rms change of the residual\nmoment about the survey mean (%)")
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        for a in (ax, bx):
            a.xaxis.set_minor_formatter(NullFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
            a.yaxis.set_major_formatter(fmt)
            a.grid(True, which="major", color=st.GRID, lw=0.5); a.set_axisbelow(True)
        st.panel_label(ax, "A", x=-0.2, y=1.02); st.panel_label(bx, "B", x=-0.2, y=1.02)
        finish(fig, "ch6_boom")


if __name__ == "__main__":
    main()
