"""Figure 6.2: concept of operations at one outcrop, as a swimlane timeline.

Durations of the four loop passes follow from the Ch. 3 survey geometry at 1 cm/s (station counts read from
results/analysis/a06_isolated_robustness.json; ch6_data.T_LOOP1, T_LOOP3). Every other duration is an assumption
of the mission concept and is listed in ch6_data.ASSUMPTIONS. The lander reference magnetometer records
throughout.
"""
from __future__ import annotations

from matplotlib.patches import Rectangle

import ch6_data as D
from ch6_common import finish, st

FS = st.BASE_PT - 1
COL = {"result": st.TRAVERSE, "assume": "#C9C9C9"}


def main():
    lanes = D.LANES
    y = {lane: len(lanes) - 1 - i for i, lane in enumerate(lanes)}
    with st.style():
        fig = st.figure("full", 2.6)
        ax = fig.add_axes([0.17, 0.19, 0.8, 0.72])
        t = 0.0
        h = 0.66
        for lane, label, dur, kind in D.CONOPS:
            yy = y[lane]
            ax.add_patch(Rectangle((t, yy - h / 2), dur, h, fc=COL[kind], ec=st.INK, lw=0.5, zorder=3))
            if dur >= 14:
                ax.text(t + dur / 2, yy, label, ha="center", va="center", fontsize=FS - 1.5, zorder=4,
                        linespacing=0.95)
            else:
                ax.text(t + dur / 2, yy + h / 2 + 0.05, label.replace("\n", " "), ha="center", va="bottom",
                        fontsize=FS - 1.5, zorder=4)
            t += dur
        total = t
        # magnetometer also records during the calibration turns (rover-field calibration data)
        t_cal0 = sum(c[2] for c in D.CONOPS[:2])
        ax.add_patch(Rectangle((t_cal0, y["Magnetometer"] - h / 2), D.CONOPS[2][2], h, fc="white", ec=st.INK,
                               lw=0.5, ls=(0, (2, 1.5)), zorder=3))
        ax.text(t_cal0 + D.CONOPS[2][2] / 2, y["Magnetometer"], "cal.\ndata", ha="center", va="center",
                fontsize=FS - 1.5, zorder=4, linespacing=0.95)
        # lander: continuous reference
        ax.add_patch(Rectangle((0, y["Lander"] - h / 2), total, h, fc=st.INTENSITY, ec=st.INK, lw=0.5, zorder=3))
        ax.text(total / 2, y["Lander"], "reference fluxgate records continuously (external field)", color="white",
                ha="center", va="center", fontsize=FS - 1, zorder=4)
        # loop-driving bracket
        t_l0 = sum(c[2] for c in D.CONOPS[:3])
        t_l1 = sum(c[2] for c in D.CONOPS[:8])
        yb = -0.75
        ax.annotate("", xy=(t_l0, yb), xytext=(t_l1, yb),
                    arrowprops=dict(arrowstyle="<->", color=st.INK, lw=0.6, shrinkA=0, shrinkB=0))
        ax.text((t_l0 + t_l1) / 2, yb - 0.08,
                f"loops and move-out: {t_l1 - t_l0:.0f} min (loop driving {D.T_TWO_REV:.0f} min)",
                ha="center", va="top", fontsize=FS - 1)
        ax.set_yticks([y[l] for l in lanes]); ax.set_yticklabels(lanes)
        ax.set_ylim(-1.35, len(lanes) - 0.35)
        ax.set_xlim(0, total * 1.005)
        ax.set_xlabel(f"Time at one outcrop (min); total {total:.0f} min = {total / 60:.1f} h")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=st.GRID, lw=0.5); ax.set_axisbelow(True)
        # legend
        lx = [Rectangle((0, 0), 1, 1, fc=COL["result"], ec=st.INK, lw=0.5),
              Rectangle((0, 0), 1, 1, fc=COL["assume"], ec=st.INK, lw=0.5)]
        ax.legend(lx, ["duration from the Ch. 3 survey geometry at 1 cm/s", "assumed duration"],
                  loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2, frameon=False, fontsize=FS - 1,
                  handlelength=1.2)
        finish(fig, "ch6_conops")


if __name__ == "__main__":
    main()
