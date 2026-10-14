"""Geometry, survey paths and SimPEG mappings shared by the simulation scripts.

    generate_traverse                  drive-around and drive-by sensor paths around an outcrop
    generate_rille_traverse            straight traverse along a wall
    create_shape_mask                  cells inside a rect, cylinder, sphere, hemisphere or pyramid
    MagnetizedShapes, MagnetizedBlocks SimPEG maps from a few uniform bodies to a vector model
    generate_pose_errors,     rigid mis-pose of the imaged outcrop (pointing and range errors)
    pose_transform

Relocated from ``engine/core.py`` (release v1.0.0); type hints added, arithmetic unchanged. ``engine/core.py`` is an
alias of this module.

Frame (thesis Section 3.3.1, "Coordinate System"): right-handed local Cartesian axes, x and y horizontal, z up; inclination is positive
downward. ``generate_traverse`` returns stations in driving order with constant sensor height.

Known discretization details of the stored runs (thesis review finding Y6-21): ``full_orbit`` loops sample each side
with ``np.linspace`` including both corners, so three corners appear twice and the loop ends one step short of its
start; these are part of the released, seeded geometry and are kept for exact reproducibility.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse
from simpeg import maps

mu0 = 4.0 * np.pi * 1e-7  # permeability of free space (H/m)

# default imaging mis-pose: none
POSE_POINTING_DEG = (0.0, 0.0)
POSE_TRANS_M = (0.0, 0.0, 0.0)


# ============================================================
# --- Survey & Synthetic Data ---
# ============================================================

def generate_traverse(traverse_type: str, shape_params: Mapping[str, float], sensor_height: float = 0.5,
                      spacing: float = 1, orbit_offset: float = 1.0, approach_distance: float = 5.0,
                      shape: str = 'rect') -> np.ndarray:
    """
    Generate survey_xyz for a selected traverse type around the configured outcrop.

    Parameters
    ----------
    traverse_type : str
        One of: "drive_by", "full_orbit", "half_orbit", "approach_square", "skate_park"
    shape_params : dict
        SHAPE_PARAMS; must include 'cx', 'cy' and shape-specific dimensions
    sensor_height : float
        Sensor height above ground (m). Used as constant z for all samples.
    spacing : float
        Nominal sample spacing along path (m).
    orbit_offset : float
        Extra clearance from the outcrop bounding radius (m).
    approach_distance : float
        Distance before/after the outcrop extent for approach/drive-by legs (m).
    shape : str, default='rect'
        Geometry type for the bounding-radius calculation.

    Returns
    -------
    survey_xyz : ndarray, shape (N, 3)
        Survey points in the same frame as (cx, cy).
    """

    # --- Extract outcrop center (XY) ---
    cx = shape_params.get("cx", 0.0)
    cy = shape_params.get("cy", 0.0)

    # --- Compute effective bounding radius (largest XY extent) ---
    if shape == "rect":
        bounding_radius = max(shape_params["sx"], shape_params["sy"]) / 2.0
    elif shape in ["cylinder_z", "sphere", "hemisphere_z+"]:
        bounding_radius = shape_params["r"]
    elif shape == "pyramid_z+":
        bounding_radius = shape_params["a"] / np.sqrt(2.0)  # center -> corner distance
    else:
        # Fallback: try sx/sy if present, otherwise default.
        bounding_radius = shape_params.get("sx", 1.0) / 2.0

    # --- TRAVERSE TYPES ---
    if traverse_type == "drive_by":
        # Straight line parallel to x-axis, offset in -y from outcrop edge.
        left_edge_x = cx - bounding_radius
        right_edge_x = cx + bounding_radius

        start_x = left_edge_x - approach_distance
        end_x = right_edge_x + approach_distance
        traverse_y = cy - bounding_radius - orbit_offset

        num_pts = max(1, int(np.ceil((end_x - start_x) / spacing)) + 1)
        traverse_x = np.linspace(start_x, end_x, num_pts)
        traverse_xy = np.column_stack([traverse_x, np.full_like(traverse_x, traverse_y)])

    elif traverse_type == "full_orbit":
        # Square orbit around outcrop, CCW.
        orbit_distance = bounding_radius + orbit_offset
        side_length = 2.0 * orbit_distance
        n_per_side = max(10, int(np.ceil(side_length / spacing)))

        x1 = np.linspace(-orbit_distance, orbit_distance, n_per_side)
        y1 = np.full_like(x1, -orbit_distance)

        x2 = np.full(n_per_side, orbit_distance)
        y2 = np.linspace(-orbit_distance, orbit_distance, n_per_side)

        x3 = np.linspace(orbit_distance, -orbit_distance, n_per_side)
        y3 = np.full_like(x3, orbit_distance)

        # Drop the last point to avoid duplicating the starting corner when concatenating.
        x4 = np.full(n_per_side - 1, -orbit_distance)
        y4 = np.linspace(orbit_distance, -orbit_distance, n_per_side)[:-1]

        traverse_xy = np.column_stack([
            np.concatenate([x1, x2, x3, x4]) + cx,
            np.concatenate([y1, y2, y3, y4]) + cy
        ])

    elif traverse_type == "half_orbit":
        # Three sides (U-shape), CCW from bottom edge.
        orbit_distance = bounding_radius + orbit_offset
        side_length = 2.0 * orbit_distance
        n_per_side = max(10, int(np.ceil(side_length / spacing)))

        x1 = np.linspace(-orbit_distance, orbit_distance, n_per_side)
        y1 = np.full_like(x1, -orbit_distance)

        x2 = np.full(n_per_side, orbit_distance)
        y2 = np.linspace(-orbit_distance, orbit_distance, n_per_side)

        x3 = np.linspace(orbit_distance, -orbit_distance, n_per_side)
        y3 = np.full_like(x3, orbit_distance)

        traverse_xy = np.column_stack([
            np.concatenate([x1, x2, x3]) + cx,
            np.concatenate([y1, y2, y3]) + cy
        ])

    elif traverse_type == "approach_square":
        # Approach leg, then a partial square around the outcrop.
        square_half = bounding_radius + orbit_offset
        approach_start = -approach_distance
        approach_end = -(bounding_radius + orbit_offset)

        num_approach_points = max(1, int(np.ceil(abs(approach_end - approach_start) / spacing)))
        approach_x = np.linspace(approach_start, approach_end, num_approach_points)
        approach_y = np.full_like(approach_x, cy - square_half)
        approach_xy = np.column_stack([approach_x + cx, approach_y])

        corners = np.array([
            [cx - square_half, cy - square_half],  # SW
            [cx + square_half, cy - square_half],  # SE
            [cx + square_half, cy + square_half],  # NE
            [cx - square_half, cy + square_half],  # NW
        ])

        orbit_fraction = 1.0  # [0, 1] fraction of full perimeter to traverse
        edge_vecs = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
        edge_lens = [np.linalg.norm(v) for v in edge_vecs]
        perimeter = float(sum(edge_lens))
        target_length = float(np.clip(orbit_fraction, 0.0, 1.0)) * perimeter

        square_segments = []
        remaining = target_length

        for i in range(4):
            if remaining <= 1e-12:
                break

            L = edge_lens[i]

            if remaining >= L - 1e-12:
                n_pts = max(1, int(np.ceil(L / spacing)))
                pts = np.linspace(corners[i], corners[(i + 1) % 4], n_pts, endpoint=False)
                square_segments.append(pts)
                remaining -= L
            else:
                f = remaining / L
                if f > 1e-12:
                    n_pts = max(1, int(np.ceil((f * L) / spacing)))
                    end_pt = corners[i] + f * (corners[(i + 1) % 4] - corners[i])
                    pts = np.linspace(corners[i], end_pt, n_pts, endpoint=False)
                    square_segments.append(pts)
                remaining = 0.0
                break

        square_path_xy = np.vstack(square_segments) if square_segments else np.empty((0, 2))
        traverse_xy = np.vstack([approach_xy, square_path_xy]) if square_path_xy.size else approach_xy.copy()

    elif traverse_type == "skate_park":
        # Square trajectory around primary outcrop.
        # Other outcrops (1m/3m/10m) are created elsewhere; not included in path.
        orbit_distance = bounding_radius + orbit_offset
        side_length = 2.0 * orbit_distance
        n_per_side = max(10, int(np.ceil(side_length / spacing)))

        x1 = np.linspace(-orbit_distance, orbit_distance, n_per_side)
        y1 = np.full_like(x1, -orbit_distance)

        x2 = np.full(n_per_side, orbit_distance)
        y2 = np.linspace(-orbit_distance, orbit_distance, n_per_side)

        x3 = np.linspace(orbit_distance, -orbit_distance, n_per_side)
        y3 = np.full_like(x3, orbit_distance)

        x4 = np.full(n_per_side - 1, -orbit_distance)
        y4 = np.linspace(orbit_distance, -orbit_distance, n_per_side)[:-1]

        traverse_xy = np.column_stack([
            np.concatenate([x1, x2, x3, x4]) + cx,
            np.concatenate([y1, y2, y3, y4]) + cy
        ])

    else:
        raise ValueError(
            f"Unknown traverse_type: '{traverse_type}'. "
            "Options: 'drive_by', 'full_orbit', 'half_orbit', 'approach_square', 'skate_park'"
        )

    # --- Add elevation (constant z) ---
    traverse_z = np.full(traverse_xy.shape[0], sensor_height)
    survey_xyz = np.column_stack([traverse_xy, traverse_z])

    return survey_xyz


def generate_rille_traverse(xmin: float, xmax: float, y: float, z_sensor: float, spacing: float) -> np.ndarray:
    """Straight-line traverse along +X at constant (y, z) for the rille-wall scenario."""
    x = np.arange(xmin, xmax + 0.5 * spacing, spacing, dtype=float)
    return np.column_stack([x, np.full_like(x, y), np.full_like(x, z_sensor)])


# ============================================================
# --- Forward Modeling ---
# ============================================================

def create_shape_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray, cx: float, cy: float, cz: float, shape: str,
                      shape_params: Mapping[str, float], margin_frac: float = 0.0,
                      margin_z: float = 0.0) -> np.ndarray:
    """
    Return a boolean mask of mesh cell-centers that lie inside a specified shape.

    Inputs
    ------
    x, y, z : ndarray, shape (nC,)
        Cell-center coordinates (typically mesh.cell_centers columns).
    cx, cy, cz : float
        Shape center/anchor in the local frame (meters).
        Interpretation depends on shape:
        - rect / cylinder / sphere: center of the solid
        - hemisphere_z+: center of the underlying sphere; enforced z >= cz
        - pyramid_z+: base plane is at cz (apex points +z)
    shape : str
        'rect', 'cylinder_z', 'sphere', 'hemisphere_z+', or 'pyramid_z+'
    shape_params : dict
        Shape dimensions:
        - rect: sx, sy, sz
        - cylinder_z: r, h
        - sphere / hemisphere_z+: r
        - pyramid_z+: a, h
    margin_frac : float
        Fractional inflation applied to XY extents (0.0 = none).
    margin_z : float
        Fractional inflation applied to vertical extents (0.0 = none).

    Output
    ------
    mask : ndarray(bool), shape (nC,)
        True where the cell center is inside the shape (after inflation).
    """
    if shape == 'rect':
        sx, sy, sz = shape_params['sx'], shape_params['sy'], shape_params['sz']

        # Half-widths inflated by margin_frac; vertical half-height inflated by margin_z.
        x_margin = (sx / 2.0) * (1 + margin_frac)
        y_margin = (sy / 2.0) * (1 + margin_frac)
        z_margin_low  = cz - (sz / 2.0) * (1 + margin_z)
        z_margin_high = cz + (sz / 2.0) * (1 + margin_z)

        return (
            (np.abs(x - cx) <= x_margin) &
            (np.abs(y - cy) <= y_margin) &
            (z >= z_margin_low) & (z <= z_margin_high)
        )

    elif shape == 'cylinder_z':
        r = shape_params['r']
        h = shape_params['h']

        r_margin = r * (1 + margin_frac)
        z_margin_low  = cz - (h / 2.0) * (1 + margin_z)
        z_margin_high = cz + (h / 2.0) * (1 + margin_z)

        dist_xy = np.sqrt((x - cx)**2 + (y - cy)**2)
        return (
            (dist_xy <= r_margin) &
            (z >= z_margin_low) & (z <= z_margin_high)
        )

    elif shape == 'sphere':
        r = shape_params['r']
        r_margin = r * (1 + margin_frac)

        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        return dist <= r_margin

    elif shape == 'hemisphere_z+':
        # Hemisphere is the upper half of a sphere: dist <= r and z >= cz
        r = shape_params['r']
        r_margin = r * (1 + margin_frac)

        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        return (dist <= r_margin) & (z >= cz)

    elif shape == 'pyramid_z+':
        # Square pyramid with base on plane z=cz and apex at z=cz+h (tapers linearly).
        a = shape_params['a']
        h = shape_params['h']

        # Prefer cz stored in params for pyramid base consistency (if present).
        cz = shape_params.get('cz', cz)

        z_rel = z - cz
        base_half = (a / 2.0) * (1 + margin_frac)
        h_margin = h * (1 + margin_z)

        # Linear taper: allowable half-width shrinks from base_half at z_rel=0 to 0 at z_rel=h_margin.
        taper_factor = np.where(
            (z_rel >= 0) & (z_rel <= h_margin),
            1.0 - (z_rel / h_margin),
            0.0
        )
        allowed_xy = base_half * taper_factor

        return (
            (np.abs(x - cx) <= allowed_xy) &
            (np.abs(y - cy) <= allowed_xy) &
            (z_rel >= 0) & (z_rel <= h_margin)   # above base plane only
        )

    else:
        raise ValueError(f"Unknown SHAPE: '{shape}'")


# ============================================================
# --- Inversion ---
# ============================================================


class MagnetizedShapes(maps.IdentityMap):
    """
    Shape-aware parametric mapping for a uniformly magnetized body.

    Supported shapes and parameter vectors
    --------------------------------------
    rect:
        [x1, y1, z1, x2, y2, z2, mx, my, mz]
        - Geometry is an axis-aligned rectangular prism spanning (x1,y1,z1) to (x2,y2,z2).

    hemisphere_z+:
        [cx, cy, cz, r, mx, my, mz]
        - Hemisphere of radius r centered at (cx,cy,cz), keeping only z >= cz.

    cylinder_z:
        [cx, cy, cz, r, h, mx, my, mz]
        - Vertical cylinder of radius r and height h centered at (cx,cy,cz).

    sphere:
        [cx, cy, cz, r, mx, my, mz]
        - Sphere of radius r centered at (cx,cy,cz).

    pyramid_z+:
        [cx, cy, cz, a, h, mx, my, mz]
        - Square pyramid: base a×a on plane z=cz, apex at z=cz+h (upward).

    Output convention
    -----------------
    Returns a 1D vector of length 3*nC_active:
        [Mx(active); My(active); Mz(active)]
    """

    def __init__(self, mesh, shape="rect", active_cells=None):
        self.geom_shape = shape
        self._mesh_ref = mesh

        # Number of parameters per shape (must match the ordering above)
        nP_map = {
            "rect": 9,
            "hemisphere_z+": 7,
            "cylinder_z": 8,
            "sphere": 7,
            "pyramid_z+": 8,  # cx, cy, cz, a, h, mx, my, mz
        }
        nP = nP_map.get(shape, 9)

        if active_cells is None:
            self.active_cells = np.ones(mesh.nC, dtype=bool)
        else:
            self.active_cells = np.asarray(active_cells, dtype=bool)

        self.nC_active = int(self.active_cells.sum())
        super().__init__(nP=nP)

    def _active_centers(self):
        """Convenience: active cell centers as (nC_active, 3)."""
        return self._mesh_ref.cell_centers[self.active_cells]

    def _inside_mask(self, params):
        """
        Compute boolean mask (length nC_active) for active cells inside the shape.

        Notes
        -----
        Ground cells (z < 0) are excluded for 'rect' and 'hemisphere_z+'.
        """
        centers = self._active_centers()
        x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

        if self.geom_shape == "rect":
            x1, y1, z1, x2, y2, z2 = params[:6]
            inside = (
                (x >= min(x1, x2)) & (x <= max(x1, x2)) &
                (y >= min(y1, y2)) & (y <= max(y1, y2)) &
                (z >= min(z1, z2)) & (z <= max(z1, z2)) &
                (z >= 0.0)
            )
            return inside

        if self.geom_shape == "hemisphere_z+":
            cx, cy, cz, r = params[:4]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            inside = (dist <= r) & (z >= cz) & (z >= 0.0)
            return inside

        if self.geom_shape == "cylinder_z":
            cx, cy, cz, r, h = params[:5]
            dist_xy = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            inside = (dist_xy <= r) & (z >= cz - h / 2) & (z <= cz + h / 2)
            return inside

        if self.geom_shape == "sphere":
            cx, cy, cz, r = params[:4]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            inside = dist <= r
            return inside

        if self.geom_shape == "pyramid_z+":
            cx, cy, cz, a, h = params[:5]
            z_rel = z - cz
            base_half = a / 2.0
            taper_factor = np.where(
                (z_rel >= 0) & (z_rel <= h),
                1.0 - (z_rel / h),
                0.0,
            )
            allowed_xy = base_half * taper_factor
            return (
                (np.abs(x - cx) <= allowed_xy)
                & (np.abs(y - cy) <= allowed_xy)
                & (z_rel >= 0)
                & (z_rel <= h)
                & (z >= 0.0)
            )

        raise ValueError(f"Unknown shape: {self.geom_shape}")

    def _transform(self, m):
        """
        Convert shape parameters -> flattened active-cell vector magnetization model.

        Parameters
        ----------
        m : array-like
            Shape parameter vector in the ordering defined by the docstring.

        Returns
        -------
        m_vec : ndarray
            Flattened vector magnetization on active cells:
            [Mx(active); My(active); Mz(active)]
        """
        m = np.asarray(m, dtype=float)
        inside = self._inside_mask(m)

        # Allocate active-cell magnetization (uniform inside, 0 outside)
        M = np.zeros((self.nC_active, 3), dtype=float)

        # Extract magnetization components from the tail of the parameter vector
        if self.geom_shape == "rect":
            mx, my, mz = m[6:9]
        elif self.geom_shape == "cylinder_z":
            mx, my, mz = m[5:8]
        elif self.geom_shape in ("hemisphere_z+", "sphere"):
            mx, my, mz = m[4:7]
        elif self.geom_shape == "pyramid_z+":
            mx, my, mz = m[5:8]
        else:
            raise ValueError(f"Unknown shape: {self.geom_shape}")

        M[inside, :] = (mx, my, mz)

        # SimPEG "vector model" convention: stack components
        return np.concatenate([M[:, 0], M[:, 1], M[:, 2]])

    def _deriv_matrix(self, m):
        """
        Jacobian for magnetization parameters only (mx,my,mz), with geometry frozen.

        This is intentionally a sparse matrix of shape:
            (3*nC_active, nP)

        The geometry parameters are treated as constant here; their derivatives
        are not included.
        """
        m = np.asarray(m, dtype=float)
        inside = self._inside_mask(m)

        n_out = 3 * self.nC_active
        J = scipy.sparse.lil_matrix((n_out, self.nP), dtype=float)

        # Where do (mx,my,mz) start in the parameter vector?
        if self.geom_shape == "rect":
            mag_idx = 6
        elif self.geom_shape == "cylinder_z":
            mag_idx = 5
        elif self.geom_shape in ("hemisphere_z+", "sphere"):
            mag_idx = 4
        elif self.geom_shape == "pyramid_z+":
            mag_idx = 5
        else:
            raise ValueError(f"Unknown shape: {self.geom_shape}")

        idx = np.where(inside)[0]
        for cell in idx:
            J[cell, mag_idx + 0] = 1.0                       # dMx/dmx
            J[cell + self.nC_active, mag_idx + 1] = 1.0      # dMy/dmy
            J[cell + 2 * self.nC_active, mag_idx + 2] = 1.0  # dMz/dmz

        return J.tocsr()

    def deriv(self, m, v=None):
        """
        SimPEG API: return J or J@v if a vector v is provided.
        """
        J = self._deriv_matrix(m)
        return J @ v if v is not None else J


class MagnetizedBlocks(maps.IdentityMap):
    """
    Parametric mapping that defines one or more magnetized rectangular blocks
    on a given mesh. Each block has 9 parameters:
        [x1, y1, z1, x2, y2, z2, mx, my, mz]
    The mapping converts these block parameters into a full 3-component
    magnetization vector on the active mesh cells.
    """

    def __init__(self, mesh, n_blocks=1, active_cells=None):
        super().__init__(nP=n_blocks*9)
        self._mesh_ref = mesh
        self.n_blocks = n_blocks

        # Handle active region (exclude air if mask provided)
        if active_cells is None:
            self.active_cells = np.ones(mesh.nC, dtype=bool)
        else:
            self.active_cells = np.asarray(active_cells, dtype=bool)

        self.nC_active = int(self.active_cells.sum())
        nP = n_blocks * 9  # 6 geom + 3 magnetization params per block
        super().__init__(nP=nP)

    # ------------------------------------------------------------------
    # Forward transform: parameters → full magnetization vector
    # ------------------------------------------------------------------
    def _transform(self, m):
        """
        Convert parametric block parameters into (nC_active * 3,) magnetization.
        """
        m = np.asarray(m, dtype=float)
        M = np.zeros((self.nC_active, 3))
        centers = self._mesh_ref.cell_centers[self.active_cells]

        for i in range(self.n_blocks):
            base = i * 9
            if base + 9 > len(m):
                break

            # Unpack block params
            x1, y1, z1, x2, y2, z2, mx, my, mz = m[base:base + 9]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            z_min, z_max = sorted([z1, z2])

            inside = (
                (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) &
                (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max) &
                (centers[:, 2] >= z_min) & (centers[:, 2] <= z_max)
            )

            # Assign uniform magnetization within this block
            M[inside, 0] = mx
            M[inside, 1] = my
            M[inside, 2] = mz

        # Return in SimPEG vector format (x-block, y-block, z-block)
        return np.concatenate([M[:, 0], M[:, 1], M[:, 2]])

    # ------------------------------------------------------------------
    # Jacobian (derivative) approximation — magnetization terms only
    # ------------------------------------------------------------------
    def _deriv_matrix(self, m):
        """
        Sparse Jacobian approximation for derivative of mapping.
        Only magnetization parameters (mx,my,mz) have direct derivatives of 1
        inside each block. Geometry derivatives are set to zero for now.
        """
        n_out = self.nC_active * 3
        nP = self.n_blocks * 9
        J = scipy.sparse.lil_matrix((n_out, nP))

        centers = self._mesh_ref.cell_centers[self.active_cells]
        for i in range(self.n_blocks):
            base = i * 9
            if base + 9 > len(m):
                break

            x1, y1, z1, x2, y2, z2, *_ = m[base:base + 9]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            z_min, z_max = sorted([z1, z2])

            inside = (
                (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) &
                (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max) &
                (centers[:, 2] >= z_min) & (centers[:, 2] <= z_max)
            )
            idx = np.where(inside)[0]

            # Fill magnetization derivatives (geometry derivatives omitted)
            for cell in idx:
                J[cell, base + 6] = 1.0               # ∂Mx/∂mx
                J[cell + self.nC_active, base + 7] = 1.0  # ∂My/∂my
                J[cell + 2*self.nC_active, base + 8] = 1.0  # ∂Mz/∂mz

        return J.tocsr()

    def deriv(self, m, v=None):
        J = self._deriv_matrix(m)
        return J @ v if v is not None else J


def generate_pose_errors(randomize: bool = False,
                         pointing_err_max_deg: Sequence[float] = (1, 1),
                         range_err_max_m: float = 0.01) -> tuple[tuple[float, float], tuple[float, ...]]:
    """
    Generate a single realization of the imaging pose error.

    Parameters
    ----------
    randomize : bool, optional
        If True, sample uniformly within specified bounds.
        If False, enforce worst-case magnitudes with random directions:
          - pointing errors lie on the boundary of an az/el ellipse
          - translation is a fixed-magnitude vector in a random direction
    pointing_err_max_deg : array-like length 2
        Maximum (az, el) pointing error bounds in degrees.
    range_err_max_m : float
        With ``randomize=False``: the translation magnitude (meters). With ``randomize=True``: the bound on EACH
        Cartesian component (uniform in [-range_err_max_m, +range_err_max_m]), so the total translation can reach
        sqrt(3) times this value (thesis review finding Y6, assumption 4).

    Returns
    -------
    pointing_deg : tuple[float, float]
        (az_err_deg, el_err_deg) pointing errors in degrees.
    trans_m : tuple[float, float, float]
        (tx, ty, tz) translation in meters representing range/position error.
    """
    if randomize:
        az_err = np.random.uniform(
            -pointing_err_max_deg[0],
            +pointing_err_max_deg[0],
        )
        el_err = np.random.uniform(
            -pointing_err_max_deg[1],
            +pointing_err_max_deg[1],
        )
        trans = np.random.uniform(
            -range_err_max_m,
            +range_err_max_m,
            size=3,
        )
        return (float(az_err), float(el_err)), tuple(trans)

    # Worst-case magnitudes with random direction:
    # Pointing: random direction on an az/el ellipse boundary
    max_az = float(pointing_err_max_deg[0])
    max_el = float(pointing_err_max_deg[1])
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    az_err = max_az * np.cos(phi)
    el_err = max_el * np.sin(phi)

    # Translation: random direction on unit sphere, fixed magnitude
    v = np.random.normal(size=3)
    v /= np.linalg.norm(v)
    trans = v * float(range_err_max_m)

    return (float(az_err), float(el_err)), tuple(trans)


def pose_transform(vertices: np.ndarray, pointing_deg: Sequence[float] = POSE_POINTING_DEG,
                   trans: Sequence[float] = POSE_TRANS_M) -> np.ndarray:
    """
    Apply a rigid transform that approximates imaging registration errors.

    The transform is:
        v' = R * v + T
    where R is a composition of:
        - azimuth rotation about +Z
        - elevation rotation about +Y
    and T is a 3D translation.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Input vertex coordinates.
    pointing_deg : tuple[float, float], optional
        (az_deg, el_deg) rotation angles in degrees. Defaults to the module-level
        ``POSE_POINTING_DEG`` (no-op). Pass an explicit realization from
        ``generate_pose_errors`` to apply non-trivial pointing error.
    trans : tuple[float, float, float], optional
        Translation (tx, ty, tz) in meters. Defaults to the module-level
        ``POSE_TRANS_M`` (no-op).

    Returns
    -------
    vertices_transformed : ndarray, shape (N, 3)
        Transformed vertex coordinates.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3); got {v.shape}")

    az_deg, el_deg = pointing_deg
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    # Rotation about Z (azimuth)
    R_az = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az),  np.cos(az), 0.0],
            [0.0,         0.0,        1.0],
        ],
        dtype=float,
    )

    # Rotation about Y (elevation)
    R_el = np.array(
        [
            [ np.cos(el), 0.0, np.sin(el)],
            [ 0.0,        1.0, 0.0       ],
            [-np.sin(el), 0.0, np.cos(el)],
        ],
        dtype=float,
    )

    # Apply elevation after azimuth: v' = R_el * (R_az * v)
    R = R_el @ R_az
    T = np.asarray(trans, dtype=float)

    return (v @ R.T) + T
