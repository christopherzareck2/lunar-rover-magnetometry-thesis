"""Figure 6.3: surface hours needed versus the number of flows surveyed, against one lunar day of daylight.

Per outcrop: loop driving alone (each loop driven both ways at 1 cm/s, ch6_data.T_TWO_REV, from the Ch. 3
geometry) or the full per-outcrop timeline of Fig. 6.2 (ch6_data.T_OUTCROP, which adds assumed durations).
Ten outcrops per flow; the full timeline also adds an assumed 500 m transit between flows, charged once per flow.
Capacities are reported as whole (complete) flows, with the fractional flow-equivalent in parentheses. The horizontal lines
are the daylight of one lunar day (half the 29.53 d synodic period, 354 h; NSSDC Moon fact sheet) times an assumed
fraction usable for surface operations.
"""
from __future__ import annotations

import numpy as np

import ch6_data as D
from ch6_common import finish, st

FS = st.BASE_PT - 1


def main():
    n = np.arange(0, 11)
    h_loops = n * D.OUTCROPS_PER_FLOW * D.T_OUTCROP_LOOPS / 60.0
    h_full = n * D.hours_per_flow(D.T_OUTCROP)
    with st.style():
        fig = st.figure("0.9", 2.9)
        ax = fig.add_axes([0.11, 0.15, 0.6, 0.8])
        ax.plot(n, h_full, "-o", color=st.DIRECTION, ms=3.5, lw=1.2, zorder=4,
                label=f"full timeline ({D.hours_per_flow(D.T_OUTCROP):.1f} h per flow)")
        ax.plot(n, h_loops, "--s", color=st.INTENSITY, ms=3.2, lw=1.1, zorder=4,
                label=f"loop driving only ({D.OUTCROPS_PER_FLOW * D.T_OUTCROP_LOOPS / 60:.1f} h per flow)")
        styles = {1.0: (0, ()), 0.5: (0, (4, 2.5)), 0.33: (0, (1, 1.5))}
        names = {1.0: "all daylight", 0.5: "half of daylight", 0.33: "one third of daylight"}
        for eta in D.OPS_EFFICIENCY:
            yv = eta * D.DAYLIGHT_H
            ax.axhline(yv, color=st.MUTED, lw=0.9, ls=styles[eta], zorder=2)
            nf = D.flows_per_lunar_day(D.T_OUTCROP, eta)
            ax.plot([nf], [yv], marker="o", ms=5, mfc="white", mec=st.INK, mew=0.8, zorder=5)
            ax.text(10.25, yv, f"{names[eta]}, {yv:.0f} h:\n{int(nf)} complete flows ({nf:.1f})", va="center", ha="left",
                    fontsize=FS - 0.5, color=st.INK, clip_on=False)
        ax.set_xlim(0, 10); ax.set_ylim(0, 560)
        ax.set_xticks(range(0, 11))
        ax.set_xlabel(f"Flows surveyed ({D.OUTCROPS_PER_FLOW} outcrops each)")
        ax.set_ylabel("Surface hours needed")
        ax.yaxis.grid(True, color=st.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=1.0, edgecolor="#BBBBBB")
        ax.text(10.25, 535, "One lunar day:", fontsize=FS - 0.5, ha="left", va="center", clip_on=False)
        finish(fig, "ch6_surface_time")


if __name__ == "__main__":
    main()
