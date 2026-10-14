"""Figure 6.4: trade study 1, accuracy versus survey time across survey architectures.

All accuracies are stored results (no new simulation), read by ch6_data from results/analysis:
(A) white noise only, a06 (42 magnetization directions; Table 4.2 and Sec. 4.2): site field known (open
    markers) or fitted with the magnetization (filled; the two-loop case also fits a uniform gradient);
(B) the requirement-set errors of Table 5.1 (a09 REQ_a and REQF_a; six directions, 120 draws each);
(C) two loops with a buried extent drawn in 0-1 m (a09 R_burial_root_fit; three directions, 150 draws):
    exposed-only model versus a 1 m root fitted, loops centred or 0.5 m off centre. Fitting the root costs no
    survey time, so (C) has no time axis.
Bars span the smallest to largest per-direction mean direction error; the marker is the largest (worst
direction), which is the figure of merit. Survey time is loop driving at 1 cm/s (ch6_data, Ch. 3 geometry).
"""
from __future__ import annotations

import ch6_data as D
from ch6_common import finish, st

FS = st.BASE_PT - 1


def bar(ax, x, lo, hi, color, filled=True, marker="o", ms=4.5, lw=1.4):
    ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    ax.plot([x], [hi], marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=1.0, zorder=4)


def main():
    with st.style():
        fig = st.figure("full", 2.7)
        ax = fig.add_axes([0.07, 0.17, 0.4, 0.74])
        bx = fig.add_axes([0.54, 0.17, 0.2, 0.74], sharey=ax)
        cx = fig.add_axes([0.80, 0.17, 0.18, 0.74], sharey=ax)

        # (A) white noise
        for name, t, (lo, hi), _ in D.ARCH_WHITE_KNOWN:
            bar(ax, t - 2.5, lo, hi, st.MUTED, filled=False)
        front = []
        for name, t, (lo, hi), _ in D.ARCH_WHITE_FITTED:
            bar(ax, t + 2.5, lo, hi, st.DIRECTION)
            front.append((t + 2.5, hi))
        ax.plot([p[0] for p in front], [p[1] for p in front], color=st.DIRECTION, lw=0.8, ls=(0, (3, 2)), zorder=2)
        labels = {"Drive-by": (8, 10.6, "left"), "One loop": (44, 9.6, "left"), "Loop + 10 m leg": (62, 6.1, "left"),
                  "Two loops": (109, 2.0, "center")}
        for name, t, (lo, hi), _ in D.ARCH_WHITE_FITTED:
            x, y, ha = labels[name]
            ax.text(x, y, name, fontsize=FS, ha=ha, va="bottom")
        ax.plot([], [], "o-", color=st.DIRECTION, ms=4, label="site field fitted")
        ax.plot([], [], "o-", color=st.MUTED, mfc="white", ms=4, label="site field known")
        ax.legend(loc="upper right", frameon=False, handlelength=1.4)
        ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100, 125])
        ax.set_xlabel("Survey time at 1 cm/s (min)")
        ax.set_ylabel("Mean direction error ($^{\\circ}$)")
        ax.set_title("White noise only", fontsize=st.BASE_PT, loc="left", pad=3)

        # (B) requirement-set errors
        pts = []
        for name, t, (lo, hi), _ in D.ARCH_SYSTEMATIC:
            bar(bx, t, lo, hi, st.INTENSITY, marker="s")
            pts.append((t, hi))
        bx.plot([p[0] for p in pts], [p[1] for p in pts], color=st.INTENSITY, lw=0.8, ls=(0, (3, 2)), zorder=2)
        bx.text(D.T_TWO, D.ARCH_SYSTEMATIC[0][2][1] + 0.5, "two loops,\none way", fontsize=FS, va="bottom",
                ha="center")
        bx.text(D.T_TWO_REV, D.ARCH_SYSTEMATIC[1][2][0] - 0.3, "both ways,\ntilt terms", fontsize=FS,
                va="top", ha="center")
        bx.set_xlim(60, 260); bx.set_xticks([100, 150, 200])
        bx.set_xlabel("Survey time (min)")
        bx.set_title("Requirement set", fontsize=st.BASE_PT, loc="left", pad=3)
        bx.tick_params(labelleft=False)

        # (C) root fit versus centring, two loops, b in 0-1 m
        xs = {"centred": 0.0, "off0.5": 1.0}
        for key, x in xs.items():
            lo, hi = D.ARCH_ROOT[key]["exposed"]; bar(cx, x - 0.13, lo, hi, st.MUTED, filled=False, marker="D", ms=4)
            lo, hi = D.ARCH_ROOT[key]["root"]; bar(cx, x + 0.13, lo, hi, st.OKABE_ITO["green"], marker="D", ms=4)
        cx.plot([], [], "D", color=st.MUTED, mfc="white", ms=4, label="exposed only")
        cx.plot([], [], "D", color=st.OKABE_ITO["green"], ms=4, label="1 m root fitted")
        cx.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False, handlelength=0.8)
        cx.set_xlim(-0.5, 1.5); cx.set_xticks([0, 1]); cx.set_xticklabels(["centered", "0.5 m off"])
        cx.set_xlabel("Loop centering")
        cx.set_title("Burial $b \\leq 1$ m", fontsize=st.BASE_PT, loc="left", pad=3)
        cx.tick_params(labelleft=False)

        for a in (ax, bx, cx):
            st.goal_line(a, 5.0, label=None)
            a.yaxis.grid(True, color=st.GRID, lw=0.5); a.set_axisbelow(True)
        ax.text(1.5, 5.05, "$5^{\\circ}$ goal", fontsize=FS, color=st.MUTED, va="bottom")
        ax.set_ylim(0, 16)
        st.panel_label(ax, "A", x=-0.12, y=1.03); st.panel_label(bx, "B", x=-0.08, y=1.03)
        st.panel_label(cx, "C", x=-0.08, y=1.03)
        finish(fig, "ch6_pareto")


if __name__ == "__main__":
    main()
