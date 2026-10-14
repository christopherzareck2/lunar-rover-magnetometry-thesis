"""Skate park: the isolated outcrop with four magnetized neighbors (paper Section 4.3, Figures 2 and 6).

Scene: the 4 x 4 x 1 m central prism of Section 4.2 plus four neighbors placed a distance
Delta_r beyond the rover traverse: a cylinder (north), a hemisphere (east), a pyramid (south)
and a sphere (west), sized as fixed fractions of the central volume. All bodies carry
|M| = 2.4e-3 A/m. Case A: every body magnetized along -z. Case B: the east and west neighbors
reversed (+z).

Survey: square traverse at 1 m standoff from the central prism, sensor at z = 0.5 m (the bodies
stand on z = 0), 1 cm sampling, three-component receivers, sigma_B = 0.50 nT per axis.

Forward model: SimPEG integral-equation simulation (Choclo engine) on a 0.25 m tensor mesh
containing all five bodies. Inversion: parametric single-body solve for the central prism only
(geometry fixed, Mx/My/Mz free). With the geometry fixed the problem is linear, so the converged
iterative solution (MagnetizedShapesRect + ProjectedGNCG) equals the closed-form weighted
least-squares solution; the script checks this once and then uses the closed-form solve for
N = 250 noise realizations at each Delta_r in 1..10 m.

Run:  python engine/skatepark_1cm_dr_sweep.py
Out:  results/engine/skatepark_1cm_dr_sweep_results.json
"""
import os
import io
import json
import contextlib

import numpy as np
import scipy.sparse as sps
import discretize
from discretize import TensorMesh
from simpeg import (maps, data, data_misfit, directives, inverse_problem,
                    optimization, inversion, regularization, utils)
from simpeg.potential_fields import magnetics as mag

# --------------------------------------------------------------------------- #
# 1) Scene constants
# --------------------------------------------------------------------------- #
SHAPE = "rect"
SHAPE_RADIUS = 2.0
SHAPE_HEIGHT = 1.0
outcrop_width = 2.0 * SHAPE_RADIUS            # = 4.0 m full width
ROVER_HEIGHT = 0.5
TRAVERSE_OFFSET = 1.0
SP_MESH_PAD_XY_M = 2.0
SP_MESH_PAD_Z_M = 0.5

SHAPE_PARAMS = {
    "cx": 0.0, "cy": 0.0,
    "sx": outcrop_width, "sy": outcrop_width, "sz": SHAPE_HEIGHT,
    "r": SHAPE_RADIUS, "h": SHAPE_HEIGHT, "a": outcrop_width,
}
SHAPE_PARAMS["cz"] = SHAPE_PARAMS["sz"] / 2.0   # rect: center at half-height (= 0.5)
cube_height = SHAPE_PARAMS["sz"]

strengthAm = 0.0024                              # A/m
inclination_deg = 90.0
declination_deg = 0.0
noise_floor = 0.5                                # nT, 1-sigma additive
relative_error = 0.0
dh_fine = 0.25                                   # m

# Imaging pointing/range uncertainty: only used to inflate the central-body mask by a small margin.
POSE_RANGE_ERR_MAX_M = 0.01
POSE_POINTING_ERR_MAX_DEG = (1.0, 1.0)

mu0 = 4.0 * np.pi * 1e-7
strengthnT = strengthAm * mu0 * 1e9              # A/m -> nT  (~3.0159 nT)

# Central-mask inflation (worst-case rotation, no translation term).
_max_rot_deg = max(abs(POSE_POINTING_ERR_MAX_DEG[0]), abs(POSE_POINTING_ERR_MAX_DEG[1]))
MASK_MARGIN_FRAC = 0.00 + 0.01 * _max_rot_deg + 0.5 * 0.0
MASK_MARGIN_Z = 0.00 + 0.005 * _max_rot_deg + 0.25 * 0.0

# Sweep controls
SPACING = 0.01                                   # 1 cm sampling
DELTA_R = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_MC = 250                                        # >= 200 draws per Delta_r
MASTER_SEED = 20260812

# True central-body magnetization vector (nT), I = 90 deg, D = 0 -> straight down (-z).
I_rad = np.deg2rad(inclination_deg)
D_rad = np.deg2rad(declination_deg)
mag_vector_true = np.array([
    strengthnT * np.cos(I_rad) * np.sin(D_rad),
    strengthnT * np.cos(I_rad) * np.cos(D_rad),
    -strengthnT * np.sin(I_rad),
])


# --------------------------------------------------------------------------- #
# 2) Shape masks
# --------------------------------------------------------------------------- #
def create_shape_mask(x, y, z, cx, cy, cz, shape, shape_params,
                      margin_frac=0.0, margin_z=0.0):
    if shape == "rect":
        sx, sy, sz = shape_params["sx"], shape_params["sy"], shape_params["sz"]
        xm = (sx / 2.0) * (1 + margin_frac)
        ym = (sy / 2.0) * (1 + margin_frac)
        zlo = cz - (sz / 2.0) * (1 + margin_z)
        zhi = cz + (sz / 2.0) * (1 + margin_z)
        return (np.abs(x - cx) <= xm) & (np.abs(y - cy) <= ym) & (z >= zlo) & (z <= zhi)
    if shape == "cylinder_z":
        r, h = shape_params["r"], shape_params["h"]
        rm = r * (1 + margin_frac)
        zlo = cz - (h / 2.0) * (1 + margin_z)
        zhi = cz + (h / 2.0) * (1 + margin_z)
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return (d <= rm) & (z >= zlo) & (z <= zhi)
    if shape == "sphere":
        r = shape_params["r"] * (1 + margin_frac)
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        return d <= r
    if shape == "hemisphere_z+":
        r = shape_params["r"] * (1 + margin_frac)
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        return (d <= r) & (z >= cz)
    if shape == "pyramid_z+":
        a, h = shape_params["a"], shape_params["h"]
        cz = shape_params.get("cz", cz)
        z_rel = z - cz
        base_half = (a / 2.0) * (1 + margin_frac)
        h_margin = h * (1 + margin_z)
        taper = np.where((z_rel >= 0) & (z_rel <= h_margin), 1.0 - (z_rel / h_margin), 0.0)
        allowed = base_half * taper
        return (np.abs(x - cx) <= allowed) & (np.abs(y - cy) <= allowed) & (z_rel >= 0) & (z_rel <= h_margin)
    raise ValueError(f"Unknown shape: {shape}")


# --------------------------------------------------------------------------- #
# 3) Neighbor layout and scene bounds
# --------------------------------------------------------------------------- #
def secondary_objects_for():
    """The four neighbors, sized as volume fractions of the 4x4x1 m central prism."""
    primary_radius = max(SHAPE_PARAMS["sx"], SHAPE_PARAMS["sy"]) / 2.0     # 2.0
    primary_volume = SHAPE_PARAMS["sx"] * SHAPE_PARAMS["sy"] * SHAPE_PARAMS["sz"]  # 16.0
    primary_height = SHAPE_PARAMS["sz"]                                    # 1.0

    r_cyl = np.sqrt((primary_volume * 0.5) / (np.pi * (primary_height * 1.0)))
    r_hemi = ((primary_volume * 0.45) * 3 / (2 * np.pi)) ** (1 / 3)
    h_pyr = primary_height * 1.2
    a_pyr = np.sqrt(3 * (primary_volume * 0.4) / h_pyr)
    r_sph = ((primary_volume * 0.3) * 3 / (4 * np.pi)) ** (1 / 3)

    return primary_radius, primary_volume, primary_height, [
        {"shape": "cylinder_z", "params": {"r": r_cyl, "h": primary_height * 1.0}, "side": "north"},
        {"shape": "hemisphere_z+", "params": {"r": r_hemi}, "side": "east"},
        {"shape": "pyramid_z+", "params": {"a": a_pyr, "h": h_pyr}, "side": "south"},
        {"shape": "sphere", "params": {"r": r_sph}, "side": "west"},
    ]


def obj_radius_of(shape_type, pr):
    if shape_type == "rect":
        return max(pr.get("sx", 0.5), pr.get("sy", 0.5)) / 2
    if shape_type in ("cylinder_z", "sphere", "hemisphere_z+"):
        return pr["r"]
    if shape_type == "pyramid_z+":
        return pr["a"] / np.sqrt(2)
    return 0.5


def outcrop_configs_for(delx, case):
    """Return the five outcrop configs (central + 4 neighbours) with per-case magnetization.

    case 'A' (parallel):     central -z, all neighbours -z.
    case 'B' (antiparallel): central -z, N/S -z, E/W +z.
    """
    primary_radius, primary_volume, primary_height, secondaries = secondary_objects_for()
    traverse_distance = primary_radius + TRAVERSE_OFFSET       # 3.0

    mag_down = mag_vector_true.copy()                          # -z
    mag_up = mag_vector_true.copy(); mag_up[2] = -mag_up[2]    # +z

    cx, cy, cz = 0.0, 0.0, SHAPE_PARAMS["cz"]
    configs = [{
        "shape": SHAPE, "center": (cx, cy, cz),
        "params": dict(SHAPE_PARAMS), "label": "Primary (rect)",
        "magnetization": mag_down,       # central is -z in both cases (global sign fixed)
        "is_primary": True,
    }]

    for obj in secondaries:
        shape_type = obj["shape"]
        params = dict(obj["params"])
        side = obj["side"]
        orad = obj_radius_of(shape_type, params)
        offset = traverse_distance + delx + orad
        if side == "north":
            ocx, ocy = 0.0, offset
        elif side == "south":
            ocx, ocy = 0.0, -offset
        elif side == "east":
            ocx, ocy = offset, 0.0
        else:  # west
            ocx, ocy = -offset, 0.0

        if shape_type == "hemisphere_z+":
            ocz = 0.0
        elif shape_type == "sphere":
            ocz = params["r"]
        elif shape_type == "cylinder_z":
            ocz = params["h"] / 2
        elif shape_type == "pyramid_z+":
            ocz = 0.0
        else:
            ocz = 0.5
        params["cx"], params["cy"], params["cz"] = ocx, ocy, ocz

        if case == "A":
            mvec = mag_down
        else:  # case B: E/W flipped
            mvec = mag_up if side in ("east", "west") else mag_down

        configs.append({
            "shape": shape_type, "center": (ocx, ocy, ocz), "params": params,
            "label": f"{shape_type} ({side})", "magnetization": mvec, "is_primary": False,
        })
    return configs, primary_radius, primary_height


def scene_bounds(delx):
    """Axis-aligned half-extent (xy) and z_top over all five bodies."""
    configs, _pr, _ph = outcrop_configs_for(delx, "A")
    xmin = xmax = ymin = ymax = z_top = 0.0
    for cfg in configs:
        st = cfg["shape"]; pr = cfg["params"]; ocx, ocy, ocz = cfg["center"]
        if st == "rect":
            sx, sy, sz = pr["sx"], pr["sy"], pr["sz"]
            bx = (ocx - sx / 2, ocx + sx / 2, ocy - sy / 2, ocy + sy / 2, ocz - sz / 2, ocz + sz / 2)
        elif st == "cylinder_z":
            r, h = pr["r"], pr["h"]
            bx = (ocx - r, ocx + r, ocy - r, ocy + r, ocz - h / 2, ocz + h / 2)
        elif st == "sphere":
            r = pr["r"]; bx = (ocx - r, ocx + r, ocy - r, ocy + r, ocz - r, ocz + r)
        elif st == "hemisphere_z+":
            r = pr["r"]; bx = (ocx - r, ocx + r, ocy - r, ocy + r, ocz, ocz + r)
        elif st == "pyramid_z+":
            a, h = pr["a"], pr["h"]; bx = (ocx - a / 2, ocx + a / 2, ocy - a / 2, ocy + a / 2, ocz, ocz + h)
        else:
            continue
        xmin = min(xmin, bx[0]); xmax = max(xmax, bx[1])
        ymin = min(ymin, bx[2]); ymax = max(ymax, bx[3]); z_top = max(z_top, bx[5])
    half = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    return float(half), float(z_top)


def build_mesh(delx):
    """Tensor mesh (dh = 0.25 m) sized to contain all five bodies plus padding, symmetric about x = y = 0.

    v1.1 fix (thesis review finding Y6-04, 2026-10-13): the horizontal extent is rounded up to a whole number of
    0.25 m cells on EACH side of the origin, so every cell face lies on a multiple of 0.25 m and the central
    4 x 4 x 1 m prism occupies exactly x, y in [-2, 2] m, z in [0, 1] m. Release v1.0.0 (``build_mesh_v1_0``)
    rounded the cell count up but set the origin from the unrounded width, which put the central prism at
    x, y in [-2.078, 1.922] m. Cell size, padding and depth are unchanged; so are the scene, survey, seeds and N.
    """
    max_depth = 4.0
    max_height = 2.0
    half_xy, z_top = scene_bounds(delx)
    need = 2.0 * half_xy + 2.0 * SP_MESH_PAD_XY_M
    region = max(16.0, need)
    n_half = int(np.ceil(region / 2.0 / dh_fine))          # cells on each side of x = 0 (and y = 0)
    max_height = max(max_height, z_top + SP_MESH_PAD_Z_M)
    hx = [(dh_fine, 2 * n_half)]
    hy = [(dh_fine, 2 * n_half)]
    hz = [(dh_fine, int(np.ceil((max_height + max_depth) / dh_fine)))]
    return TensorMesh([hx, hy, hz], x0=(-n_half * dh_fine, -n_half * dh_fine, -max_depth))


def build_mesh_v1_0(delx):
    """The release v1.0.0 mesh, kept for provenance and for tests (central prism offset by -0.078 m in x and y)."""
    region_length_x = 16.0
    region_length_y = 16.0
    max_depth = 4.0
    max_height = 2.0
    half_xy, z_top = scene_bounds(delx)
    need = 2.0 * half_xy + 2.0 * SP_MESH_PAD_XY_M
    region_length_x = max(region_length_x, need)
    region_length_y = max(region_length_y, need)
    max_height = max(max_height, z_top + SP_MESH_PAD_Z_M)
    hx = [(dh_fine, int(np.ceil(region_length_x / dh_fine)))]
    hy = [(dh_fine, int(np.ceil(region_length_y / dh_fine)))]
    hz = [(dh_fine, int(np.ceil((max_height + max_depth) / dh_fine)))]
    x0 = (-region_length_x / 2.0, -region_length_y / 2.0, -max_depth)
    return TensorMesh([hx, hy, hz], x0=x0)


# --------------------------------------------------------------------------- #
# 4) Survey: square traverse at 1 cm
# --------------------------------------------------------------------------- #
def build_survey_xyz(spacing=SPACING):
    bounding_radius = max(SHAPE_PARAMS["sx"], SHAPE_PARAMS["sy"]) / 2.0    # 2.0
    orbit_distance = bounding_radius + TRAVERSE_OFFSET                     # 3.0
    side_length = 2.0 * orbit_distance
    n = max(10, int(np.ceil(side_length / spacing)))
    x1 = np.linspace(-orbit_distance, orbit_distance, n); y1 = np.full_like(x1, -orbit_distance)
    x2 = np.full(n, orbit_distance); y2 = np.linspace(-orbit_distance, orbit_distance, n)
    x3 = np.linspace(orbit_distance, -orbit_distance, n); y3 = np.full_like(x3, orbit_distance)
    x4 = np.full(n - 1, -orbit_distance); y4 = np.linspace(orbit_distance, -orbit_distance, n)[:-1]
    xy = np.column_stack([np.concatenate([x1, x2, x3, x4]), np.concatenate([y1, y2, y3, y4])])
    z = np.full(xy.shape[0], ROVER_HEIGHT)
    return np.column_stack([xy, z])


def make_survey(sxyz):
    rx = mag.receivers.Point(np.asarray(sxyz, float), components=["bx", "by", "bz"])
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], amplitude=1.0,
                                             inclination=0.0, declination=0.0)
    return mag.survey.Survey(src)


# --------------------------------------------------------------------------- #
# 5) Forward: build masks, clean data (all 5 bodies), central operator G_cen
# --------------------------------------------------------------------------- #
def sim_over(mesh, survey, active_cells):
    nC = int(active_cells.sum())
    return mag.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey, chiMap=maps.IdentityMap(nP=3 * nC),
        active_cells=active_cells, model_type="vector", engine="choclo")


def build_scene_operators(delx, survey):
    """Return dict with masks, clean data per case, and central operator G_cen (nD x 3)."""
    mesh = build_mesh(delx)
    cc = mesh.cell_centers
    x, y, z = cc[:, 0], cc[:, 1], cc[:, 2]

    # Per-case magnetization fields and union mask.
    def build_case(case):
        configs, _pr, _ph = outcrop_configs_for(delx, case)
        M = np.zeros((mesh.nC, 3))
        inside_all = np.zeros(mesh.nC, dtype=bool)
        primary = np.zeros(mesh.nC, dtype=bool)
        for i, cfg in enumerate(configs):
            ocx, ocy, ocz = cfg["center"]
            mask = create_shape_mask(
                x, y, z, ocx, ocy, ocz, cfg["shape"], cfg["params"],
                margin_frac=MASK_MARGIN_FRAC if cfg["is_primary"] else 0.0,
                margin_z=MASK_MARGIN_Z if cfg["is_primary"] else 0.0)
            M[mask] = cfg["magnetization"]
            inside_all |= mask
            if cfg["is_primary"]:
                primary = mask.copy()
        M[z < 0.0] = 0.0
        return M, inside_all, primary

    M_A, inside_all_A, primary = build_case("A")
    M_B, inside_all_B, _ = build_case("B")

    # Clean data over each case's union active set.
    sim_A = sim_over(mesh, survey, inside_all_A)
    d_clean_A = np.asarray(sim_A.dpred(utils.mkvc(M_A[inside_all_A])), float)
    sim_B = sim_over(mesh, survey, inside_all_B)
    d_clean_B = np.asarray(sim_B.dpred(utils.mkvc(M_B[inside_all_B])), float)

    # Central-only operator G_cen (nD x 3): the parametric inversion's forward map.
    nCc = int(primary.sum())
    sim_cen = sim_over(mesh, survey, primary)
    cols = []
    for k in range(3):
        e = np.zeros(3 * nCc)
        e[k * nCc:(k + 1) * nCc] = 1.0
        cols.append(np.asarray(sim_cen.dpred(e), float))
    G_cen = np.column_stack(cols)                 # (nD, 3), maps uniform (Mx,My,Mz)[nT] -> nT

    return {
        "mesh": mesh, "primary_mask": primary, "nC_central": nCc,
        "d_clean": {"A": d_clean_A, "B": d_clean_B},
        "n_cells_union": {"A": int(inside_all_A.sum()), "B": int(inside_all_B.sum())},
        "G_cen": G_cen, "sim_cen": sim_cen,
    }


# --------------------------------------------------------------------------- #
# 6) Metrics
# --------------------------------------------------------------------------- #
def dir_err_deg(rec, true):
    rec = np.asarray(rec, float); true = np.asarray(true, float)
    c = rec @ true / (np.linalg.norm(rec) * np.linalg.norm(true) + 1e-300)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def summarize(a):
    a = np.asarray(a, float)
    return {"median": float(np.median(a)), "mean": float(np.mean(a)), "std": float(np.std(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)), "max": float(np.max(a)), "N": int(a.size)}


# --------------------------------------------------------------------------- #
# 7) Validation: converged iterative (MagnetizedShapes) vs closed-form
# --------------------------------------------------------------------------- #
class MagnetizedShapesRect(maps.IdentityMap):
    """Parametric map for one rectangular body (9 parameters: bounding box + Mx, My, Mz);
    only Mx, My, Mz have nonzero derivatives because the geometry is frozen."""

    def __init__(self, mesh, active_cells):
        self._mesh_ref = mesh
        self.active_cells = np.asarray(active_cells, bool)
        self.nC_active = int(self.active_cells.sum())
        super().__init__(nP=9)

    def _inside(self, m):
        c = self._mesh_ref.cell_centers[self.active_cells]
        x, y, z = c[:, 0], c[:, 1], c[:, 2]
        x1, y1, z1, x2, y2, z2 = m[:6]
        return ((x >= min(x1, x2)) & (x <= max(x1, x2)) & (y >= min(y1, y2)) & (y <= max(y1, y2))
                & (z >= min(z1, z2)) & (z <= max(z1, z2)) & (z >= 0.0))

    def _transform(self, m):
        m = np.asarray(m, float)
        inside = self._inside(m)
        M = np.zeros((self.nC_active, 3))
        M[inside, :] = m[6:9]
        return np.concatenate([M[:, 0], M[:, 1], M[:, 2]])

    def deriv(self, m, v=None):
        inside = self._inside(np.asarray(m, float))
        J = sps.lil_matrix((3 * self.nC_active, 9))
        idx = np.where(inside)[0]
        for cell in idx:
            J[cell, 6] = 1.0
            J[cell + self.nC_active, 7] = 1.0
            J[cell + 2 * self.nC_active, 8] = 1.0
        J = J.tocsr()
        return J @ v if v is not None else J


def run_skateparkv3_iterative(mesh, survey, primary_mask, dobs, sigma, m0_mag):
    """One iterative parametric inversion of the central body (ProjectedGNCG)."""
    blocks_map = MagnetizedShapesRect(mesh, primary_mask)
    sim = mag.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey, chiMap=blocks_map, active_cells=primary_mask,
        model_type="vector", engine="choclo")
    dobj = data.Data(survey=survey, dobs=np.asarray(dobs, float))
    dobj.standard_deviation = np.full(int(survey.nD), float(sigma))
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=dobj)
    reg_mesh = discretize.TensorMesh([[1.0] * 9])
    reg = regularization.Smallness(mesh=reg_mesh, mapping=maps.IdentityMap(nP=9))
    reg.alpha_s = 1e-2
    opt = optimization.ProjectedGNCG(maxIter=100, lower=-1e10, upper=1e10, maxIterCG=30, tolCG=1e-5)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e-6)
    dlist = [directives.BetaEstimate_ByEig(beta0_ratio=1e-4),
             directives.UpdatePreconditioner(),
             directives.TargetMisfit(chifact=1.0)]
    inv = inversion.BaseInversion(invprob, directiveList=dlist)
    # Starting parametric model: true central bbox geometry + starting magnetization m0_mag.
    m0 = np.array([-SHAPE_RADIUS, -SHAPE_RADIUS, 0.0, SHAPE_RADIUS, SHAPE_RADIUS, cube_height,
                   *m0_mag], dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        mrec = np.asarray(inv.run(m0), float)
    return mrec[6:9]


# --------------------------------------------------------------------------- #
# 8) Main sweep
# --------------------------------------------------------------------------- #
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    sxyz = build_survey_xyz(SPACING)
    survey = make_survey(sxyz)
    nD = int(survey.nD)
    print(f"[survey] skate_park 1 cm: {len(sxyz)} stations, nD = {nD} (bx,by,bz)")

    m_true_cen = mag_vector_true.copy()             # central truth, -z, |M| = strengthnT
    true_amp_Am = strengthAm

    results = {
        "meta": {
            "scenario": "non-isolated skate park (central 4x4x1 m prism + cylinder N, hemisphere E, pyramid S, sphere W)",
            "sampling_m": SPACING, "n_stations": int(len(sxyz)), "nD": nD,
            "receiver_components": ["bx", "by", "bz"],
            "noise_floor_nT": noise_floor, "dh_fine_m": dh_fine,
            "strength_Am": strengthAm, "strength_nT": strengthnT,
            "inclination_deg": inclination_deg, "declination_deg": declination_deg,
            "traverse_offset_m": TRAVERSE_OFFSET, "rover_height_m": ROVER_HEIGHT,
            "mesh": "v1.1: 0.25 m cells, symmetric about x = y = 0; central prism exactly x, y in [-2, 2] m "
                    "(Y6-04 fix; v1.0.0 offset it to [-2.078, 1.922] m)",
            "delta_r_m": DELTA_R, "N_MC": N_MC, "master_seed": MASTER_SEED,
            "inversion": "central-body parametric (geometry frozen, Mx/My/Mz free); "
                         "converged == cold-start weighted LS (validated vs ProjectedGNCG)",
            "cases": {"A": "parallel: central -z, all neighbours -z",
                      "B": "antiparallel: central -z, N/S -z, E/W +z"},
        },
        "validation": {},
        "sweep": {"A": {}, "B": {}},
    }

    # ---- one-time validation at Delta_r = 1, case A ----
    print("\n[validate] converged iterative (MagnetizedShapes/ProjectedGNCG) vs closed-form LS ...")
    ops1 = build_scene_operators(1, survey)
    G1 = ops1["G_cen"]
    K1 = np.linalg.pinv(G1)                          # cold-start LS operator (3 x nD)
    rng_v = np.random.default_rng(MASTER_SEED)
    d_noisy = ops1["d_clean"]["A"] + rng_v.normal(0.0, noise_floor, size=nD)
    m_cf = K1 @ d_noisy
    m_it = run_skateparkv3_iterative(ops1["mesh"], survey, ops1["primary_mask"],
                                     d_noisy, noise_floor, m0_mag=m_true_cen)
    rel = float(np.linalg.norm(m_it - m_cf) / (np.linalg.norm(m_cf) + 1e-300))
    print(f"           closed-form M = {m_cf} nT")
    print(f"           iterative   M = {m_it} nT")
    print(f"           |Δ|/|M| = {rel:.4%}  dir(cf)={dir_err_deg(m_cf, m_true_cen):.3f}°  "
          f"dir(it)={dir_err_deg(m_it, m_true_cen):.3f}°")
    results["validation"] = {
        "delta_r_m": 1, "case": "A",
        "closed_form_M_nT": [float(v) for v in m_cf],
        "iterative_M_nT": [float(v) for v in m_it],
        "relative_diff": rel,
        "dir_err_closed_form_deg": dir_err_deg(m_cf, m_true_cen),
        "dir_err_iterative_deg": dir_err_deg(m_it, m_true_cen),
        "note": "closed-form cold-start LS reproduces the converged ProjectedGNCG solution; "
                "closed-form used for the N>=200 sweep",
    }

    # ---- full sweep ----
    for delx in DELTA_R:
        ops = build_scene_operators(delx, survey) if delx != 1 else ops1
        G = ops["G_cen"]
        K = np.linalg.pinv(G)
        for case in ("A", "B"):
            d_clean = ops["d_clean"][case]
            rng = np.random.default_rng(MASTER_SEED + 100 * delx + (0 if case == "A" else 1))   # deterministic per (Delta_r, case)
            noise = rng.normal(0.0, noise_floor, size=(N_MC, nD))
            D = d_clean[None, :] + noise
            M_rec = D @ K.T                          # (N, 3) recovered central M [nT]
            pred = M_rec @ G.T
            resid_rms = np.sqrt(np.mean(((pred - D) / noise_floor) ** 2, axis=1))

            de = np.array([dir_err_deg(M_rec[i], m_true_cen) for i in range(N_MC)])
            amp_Am = np.linalg.norm(M_rec, axis=1) / (mu0 * 1e9)
            me_signed = 100.0 * (amp_Am - true_amp_Am) / true_amp_Am
            me_abs = np.abs(me_signed)

            results["sweep"][case][str(delx)] = {
                "delta_r_m": delx,
                "n_cells_union": ops["n_cells_union"][case],
                "n_cells_central": ops["nC_central"],
                "dir_err_deg": summarize(de),
                "mag_err_pct_signed": summarize(me_signed),
                "mag_err_pct_abs": summarize(me_abs),
                "M_rec_Am": summarize(amp_Am),
                "residual_rms": summarize(resid_rms),
                "recovered_M_nT_mean": [float(v) for v in M_rec.mean(axis=0)],
            }
            print(f"[sweep] Δr={delx:2d} case {case}: "
                  f"dir={np.median(de):5.2f}° (p25-p75 {np.percentile(de,25):4.2f}-{np.percentile(de,75):4.2f}) | "
                  f"|M|err_signed={np.median(me_signed):6.1f}% | "
                  f"|M|={np.median(amp_Am):.5f} A/m | rms={np.median(resid_rms):.3f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "engine",
                            "skatepark_1cm_dr_sweep_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {out_path}")
    return results


if __name__ == "__main__":
    main()
