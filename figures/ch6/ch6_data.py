"""Numbers behind Chapter 6 (mission concept, requirements flowdown and trade studies).

One place for every number the Chapter 6 figures and tables use. Each value is one of three kinds, and the
comment on it says which:

* RESULT     read from the stored results in ``results/analysis`` (the same JSON files that Chapters 4 and 5
             quote), so the figure cannot drift from the tables;
* GEOMETRY   computed from the survey geometry of Chapter 3 (station counts at 1 cm spacing and the 1 cm/s speed);
* ASSUMPTION a value we chose for the mission concept. Assumptions are listed in ``ASSUMPTIONS`` so the thesis
             text and the report can quote them verbatim. Change one here and every figure follows.

Nothing in this module runs an inversion; it only reads stored results and does arithmetic.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lunarmag.paths import RES_ANALYSIS, THESIS_FIGURES  # noqa: E402

OUT_DIR = THESIS_FIGURES                      # ch6_*.pdf go next to the other thesis figures
PNG_DIR = Path(__file__).resolve().parent / "png"   # 300 dpi previews for checking

# ---------------------------------------------------------------------------------------------- constants
SPEED_M_S = 0.01            # ASSUMPTION carried from Ch 3 (survey speed 1 cm/s; Sec. 3.2.4)
SPACING_M = 0.01            # GEOMETRY: 1 cm station spacing (Sec. 3.5)
SYNODIC_DAYS = 29.53        # Moon fact sheet, NSSDC (Williams 2024): synodic period 29.53 d
DAYLIGHT_H = 0.5 * SYNODIC_DAYS * 24.0   # = 354.4 h of daylight per lunar day at a low- to mid-latitude site
VIPER_PROSPECT_M_S = 0.10   # Ch 3 (Fong 2025): ~10 cm/s while prospecting; used for transit between outcrops


def _load(name: str) -> dict:
    return json.loads((RES_ANALYSIS / name).read_text())


A06 = _load("a06_isolated_robustness.json")
A09 = _load("a09_systematics.json")


def minutes(n_stations: int, speed=SPEED_M_S) -> float:
    """Driving time of a survey of ``n_stations`` at 1 cm spacing (GEOMETRY)."""
    return (n_stations - 1) * SPACING_M / speed / 60.0


# ---------------------------------------------------------------------------------------------- survey times
N_LOOP1 = A06["I_survey_designs_for_offsets"]["loop_1m"]["n_stations"]            # 2399 (24 m loop at 1 m)
N_LEG = A06["I_survey_designs_for_offsets"]["loop_1m+10m_leg"]["n_stations"]      # 3399 (loop + 10 m leg)
N_TWO = A06["I_survey_designs_for_offsets"]["loop_1m+loop_3m"]["n_stations"]      # 6398 (loops at 1 and 3 m)
N_DRIVEBY = A06["A_dc_offsets"]["drive_by"]["n_stations"]                          # 1601 (16 m line)

T_DRIVEBY = minutes(N_DRIVEBY)          # ~26.7 min
T_LOOP1 = minutes(N_LOOP1)              # ~40.0 min
T_LEG = minutes(N_LEG)                  # ~56.6 min
T_LOOP3 = minutes(N_TWO - N_LOOP1)      # ~66.7 min (40 m loop at 3 m standoff)
T_TWO = T_LOOP1 + T_LOOP3               # ~106.7 min
T_TWO_REV = 2.0 * T_TWO                 # ~213 min, each loop driven in both directions


# ---------------------------------------------------------------------------------------------- accuracy (RESULT)
def _range(rows, key="dir_err_mean"):
    a = [r[key] for r in rows]
    return min(a), max(a)


def _a09_range(key):
    v = A09[key]
    m = [v[d]["all_sources"]["dir_err_deg"]["mean"] for d in v if d != "config"]
    return min(m), max(m)


def _root_range(dx, fit):
    R = A09["R_burial_root_fit"]
    m = [R[f"dx{dx}_bmax1_{d}"][fit]["all"]["dir_err_deg"]["mean"] for d in ("-z", "+x", "inc45")]
    return min(m), max(m)


B = A06["B_direction_height_scan"]
I = A06["I_survey_designs_for_offsets"]
H = A06["H_offsets_and_burial"]

#: Each architecture: label, survey minutes, (min, max) over magnetization directions of the mean direction
#: error (deg), condition key, source in the thesis.
ARCH_WHITE_KNOWN = [   # white noise only, constant site field known (Table 4.2 rows 1-2)
    ("Drive-by", T_DRIVEBY, _range(B["drive_by"]["height"][1]["rows"]), "Table 4.2, 42 directions"),
    ("One loop", T_LOOP1, _range(B["drive_around"]["height"][1]["rows"]), "Table 4.2, 42 directions"),
]
ARCH_WHITE_FITTED = [  # white noise only, constant site field fitted with the magnetization (Sec. 4.2)
    ("Drive-by", T_DRIVEBY, _range(H["drive_by"]["H1_world_offsets_fitted"]["rows"]), "Table 4.2"),
    ("One loop", T_LOOP1, _range(H["drive_around"]["H1_world_offsets_fitted"]["rows"]), "Table 4.2"),
    ("Loop + 10 m leg", T_LEG, _range(I["loop_1m+10m_leg"]["local_offsets"]["rows"]), "Sec. 4.2"),
    ("Two loops", T_TWO, _range(I["loop_1m+loop_3m"]["local_offsets_and_gradient"]["rows"]), "Table 4.2 (gradient also fitted)"),
]
ARCH_SYSTEMATIC = [    # requirement-set errors of Table 5.1, loops 0.1 m off centre, buried extent <= 0.5 m
    ("Two loops", T_TWO, _a09_range("REQ_a_dx0.1_b0.5"), "Table 5.1, requirement set"),
    ("Two loops, both ways\n+ tilt terms", T_TWO_REV, _a09_range("REQF_a_dx0.1_b0.5"), "Table 5.1, mitigated set"),
]
ARCH_ROOT = {          # two loops, buried extent drawn in 0-1 m, white noise (Sec. 4.2): root fitted or not
    "centred": {"exposed": _root_range(0, "exposed_only"), "root": _root_range(0, "with_root_1m")},
    "off0.5": {"exposed": _root_range(0.5, "exposed_only"), "root": _root_range(0.5, "with_root_1m")},
}

# ---------------------------------------------------------------------------------------------- ConOps (ASSUMPTION)
#: Per-outcrop activity list: (lane, label, minutes, kind). kind "result" = time follows from the Ch 3 geometry
#: at 1 cm/s; "assume" = an assumption of the mission concept.
LANES = ["Lander", "Rover mobility", "Imaging", "Magnetometer", "Susceptibility probe"]
CONOPS = [
    ("Rover mobility", "Transit", 50.0 / VIPER_PROSPECT_M_S / 60.0, "assume"),
    ("Imaging", "Approach,\nimaging", 30.0, "assume"),
    ("Rover mobility", "Calib.\nturns", 15.0, "assume"),
    ("Magnetometer", "1 m loop", T_LOOP1, "result"),
    ("Magnetometer", "1 m loop,\nreversed", T_LOOP1, "result"),
    ("Rover mobility", "Move out", 3.0, "assume"),
    ("Magnetometer", "3 m loop", T_LOOP3, "result"),
    ("Magnetometer", "3 m loop,\nreversed", T_LOOP3, "result"),
    ("Susceptibility probe", "Read-\nings", 15.0, "assume"),
    ("Magnetometer", "Base-\nline", minutes(1001), "assume"),
]
T_OUTCROP = sum(c[2] for c in CONOPS)                    # minutes per outcrop, all activities
T_OUTCROP_LOOPS = T_TWO_REV                              # minutes per outcrop, loop driving only
OUTCROPS_PER_FLOW = 10                                   # Ch 1 / Sec. 3.4.6: ~10 outcrops per flow mean
T_BETWEEN_FLOWS_H = 500.0 / VIPER_PROSPECT_M_S / 3600.0  # ASSUMPTION: 500 m between flows at 10 cm/s
OPS_EFFICIENCY = (1.0, 0.5, 0.33)                        # ASSUMPTION: fraction of daylight usable for surface ops


def hours_per_flow(t_outcrop_min: float, n_outcrops: int = OUTCROPS_PER_FLOW) -> float:
    return n_outcrops * t_outcrop_min / 60.0 + T_BETWEEN_FLOWS_H


def flows_per_lunar_day(t_outcrop_min: float, eta: float, n_outcrops: int = OUTCROPS_PER_FLOW) -> float:
    return eta * DAYLIGHT_H / hours_per_flow(t_outcrop_min, n_outcrops)


# ---------------------------------------------------------------------------------------------- Fisher statistics
def kappa_from_mean_angle(mean_deg: float) -> float:
    """Fisher precision parameter whose mean angular deviation is ``mean_deg`` (small-angle: mean = sqrt(pi/2k))."""
    th = math.radians(mean_deg)
    return math.pi / (2.0 * th * th)


def alpha95(mean_deg: float, n: int) -> float:
    """Approximate 95% cone (deg) of the mean of n Fisher-distributed directions, 140 / sqrt(k n)."""
    return 140.0 / math.sqrt(kappa_from_mean_angle(mean_deg) * n)


# ---------------------------------------------------------------------------------------------- boom (ASSUMPTION)
R0_M = 1.0                 # ASSUMPTION: distance from the rover's magnetic centroid to the sensor, baseline boom
B_RES_R0_NT = 0.40         # Table 3.1: rover AC+DC field knowledge, 0.40 nT per axis (1 sigma) at the baseline
DRIFT_REQ_NT = 0.01        # Sec. 9.2 item 2: slow drift, 0.01 nT rms about the survey mean (pooled over axes)
DRIFT_ROVER_SHARE_NT = 0.005  # Table 6.4 allocation to the rover-moment variation (ASSUMPTION)
MU0_4PI = 1e-7


def dipole_moment_for(b_axis_rms_nt: float, r_m: float, factor: float = 1.0) -> float:
    """Equivalent residual dipole moment (A m^2) whose field at r has the given per-axis rms (the rms of the three
    components of one vector, |B|/sqrt(3)); factor 1 = sensor on the dipole's equator, 2 = on its axis."""
    b_mag = b_axis_rms_nt * math.sqrt(3.0)
    return b_mag * 1e-9 * r_m ** 3 / (MU0_4PI * factor)


def residual_field(r_m, r0=R0_M, b0=B_RES_R0_NT):
    """Per-axis rms of the equivalent residual dipole field at distance r along a fixed direction, anchored at
    (r0, b0). Every component of a dipole field along a fixed direction scales as r^-3, so the per-axis rms does."""
    return b0 * (r0 / r_m) ** 3


ASSUMPTIONS = {
    "speed": "Survey speed 1 cm/s (carried from Ch 3).",
    "daylight": f"One lunar day of daylight = half the synodic period = {DAYLIGHT_H:.0f} h (NSSDC).",
    "transit_outcrops": "Transit between outcrops of one flow: 50 m at 10 cm/s (8.3 min).",
    "transit_flows": "Transit between flows: 500 m at 10 cm/s (1.4 h).",
    "imaging": "Approach and stereo/LiDAR imaging of each outcrop: 30 min.",
    "calibration": "Rover calibration turns away from the outcrop: 15 min.",
    "move_out": "Repositioning from the 1 m loop to the 3 m loop: 3 min.",
    "susceptibility": "Susceptibility probe contact readings: 15 min.",
    "baseline": "Departure baseline traverse: 10 m at 1 cm/s (16.7 min).",
    "efficiency": "Fraction of daylight usable for surface operations: 1, 0.5 or 0.33 (unknown; a sensitivity).",
    "outcrops": "Ten outcrops per flow.",
    "boom_r0": "Sensor 1 m from the rover's magnetic centroid at the baseline boom.",
    "boom_dipole": "The uncalibrated rover field is an equivalent residual point dipole whose field has a per-axis rms of 0.40 nT at 1 m.",
    "transit_convention": "Transit between flows is charged once per flow (F times for F flows).",
}

if __name__ == "__main__":   # print the derived numbers the chapter quotes
    print(f"daylight {DAYLIGHT_H:.1f} h")
    print(f"drive-by {T_DRIVEBY:.1f}, loop {T_LOOP1:.1f}, leg {T_LEG:.1f}, loop3 {T_LOOP3:.1f}, two {T_TWO:.1f}, "
          f"two rev {T_TWO_REV:.1f} min")
    for name, arch in (("known", ARCH_WHITE_KNOWN), ("fitted", ARCH_WHITE_FITTED), ("systematic", ARCH_SYSTEMATIC)):
        for a in arch:
            print(f"{name:10s} {a[0]!r:32s} {a[1]:6.1f} min  {a[2][0]:.2f}-{a[2][1]:.2f} deg  ({a[3]})")
    print("root", {k: {kk: tuple(round(x, 2) for x in vv) for kk, vv in v.items()} for k, v in ARCH_ROOT.items()})
    print(f"per outcrop all activities {T_OUTCROP:.1f} min ({T_OUTCROP/60:.2f} h); loops only {T_OUTCROP_LOOPS:.1f} min")
    for t, lab in ((T_OUTCROP_LOOPS, "loops only"), (T_OUTCROP, "full timeline")):
        print(lab, f"hours/flow {hours_per_flow(t):.1f}",
              "flows/day", [round(flows_per_lunar_day(t, e), 2) for e in OPS_EFFICIENCY])
    t_oneway = T_OUTCROP - T_TWO
    print("one-way full timeline", f"{t_oneway:.1f} min", [round(flows_per_lunar_day(t_oneway, e), 2) for e in OPS_EFFICIENCY])
    print("5 outcrops full", [round(flows_per_lunar_day(T_OUTCROP, e, 5), 2) for e in OPS_EFFICIENCY])
    for m in (3.2, 5.0, 10.0):
        print(f"alpha95 mean {m}: n=10 {alpha95(m, 10):.2f}, n=5 {alpha95(m, 5):.2f}, k={kappa_from_mean_angle(m):.0f}")
    print("equivalent residual dipole at 1 m for 0.4 nT per axis: equatorial %.2e, axial %.2e A m^2"
          % (dipole_moment_for(0.4, 1.0, 1), dipole_moment_for(0.4, 1.0, 2)))
    for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        b = residual_field(r)
        print(f"r={r}: field {b:.4f} nT, allowed fractional drift: whole budget {DRIFT_REQ_NT/b*100:.4f} %, "
              f"allocated share {DRIFT_ROVER_SHARE_NT/b*100:.4f} %")
    print("allocation RSS rover", math.sqrt(.25**2+.2**2+.15**2+.1**2), "remainder", math.sqrt(.4**2-(.25**2+.2**2+.15**2+.1**2)))
    print("allocation RSS drift", math.sqrt(3*.005**2+.004**2), "remainder", math.sqrt(.01**2-(3*.005**2+.004**2)))
    print("orientation share deg", math.degrees(0.005/300), "P.2 whole budget deg", math.degrees(0.01/300))
    print("wall traverse h", 2*minutes(24001)/60)
