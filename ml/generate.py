"""Simulate the training, validation and test sets for the amortized estimator (loops 0.5 m off centre).

Every set is built in chunks of CHUNK surveys; chunk j of set s uses the seed seed_for(f"ml:{s}:{j}") from
engine/inversion_helpers, so any chunk can be regenerated on its own and the result does not depend on the number
of worker processes.

Run:  python ml/generate.py [--sets name,name,...] [--workers 6]
Out:  ml/data/<set>.npz   (not tracked by git; about 350 MB in total)
"""
import os, sys, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CHUNK = 2000
DX = 0.5


def unit_dirs(rng, n):
    u = rng.normal(size=(n, 3)); return u / np.linalg.norm(u, axis=1, keepdims=True)


def P(**kw):
    p = dict(C.PRIOR_FULL); p.update(kw); return p


def N(**kw):
    p = dict(C.PRIOR_NONE); p.update(kw); return p


SINGLE = dict(fixed=True)   # thesis 'Z_single' amplitudes: drift and tilt at their tolerance, burial U(0, 1 m)
# name: (n, prior, kwargs, m_mode) ; m_mode: 'prior' | 'unit' (|M| = M_ref, random direction) | a DIRS6 key
SETS = {
    "train_full": (400_000, P(), dict(profile=False), "prior"),
    "val_full": (20_000, P(), dict(profile=False), "prior"),
    "train_none": (200_000, N(), dict(profile=False, constant_site=True), "prior"),
    "val_none": (10_000, N(), dict(profile=False, constant_site=True), "prior"),
    "test_prior": (20_000, P(), {}, "prior"),
    "test_none": (10_000, N(), dict(constant_site=True), "prior"),
    **{f"test_dir_{d}": (2_000, P(), {}, d) for d in C.DIRS6},
    **{f"test_b{b:.1f}": (1_000, P(), dict(b=b), "unit") for b in np.round(np.arange(0, 1.01, 0.1), 1)},
    **{f"test_bonly{b:.1f}": (1_000, N(), dict(b=b), "unit") for b in np.round(np.arange(0, 1.01, 0.1), 1)},
    "test_single_burial": (2_000, N(bmax=1.0, **SINGLE), {}, "unit"),
    "test_single_pose": (2_000, N(pose_deg=3.0, pose_m=0.15, **SINGLE), {}, "unit"),
    "test_single_drift": (2_000, N(drift_max=0.03, **SINGLE), {}, "unit"),
    "test_single_tilt": (2_000, N(tilt_max=0.05, site_B0=100.0, site_R=(1e6, 1e6), **SINGLE), {}, "unit"),
    "test_single_site": (2_000, N(site_B0=100.0, site_R=(530.0, 530.0), **SINGLE), {}, "unit"),
    "test_single_none": (2_000, N(), {}, "unit"),
    # the thesis combined budgets (a09 parts U and V) without induced magnetization: amplitudes fixed at tolerance
    **{f"test_U_{d}": (2_000, P(fixed=True, site_R=(530.0, 530.0)), {}, d) for d in C.DIRS3},
    **{f"test_V_{d}": (2_000, P(fixed=True, bmax=0.5, pose_deg=1.5, pose_m=0.075, drift_max=0.015, tilt_max=0.025,
                                site_R=(750.0, 750.0)), {}, d) for d in C.DIRS3},
    "test_ood_burial": (3_000, P(), dict(b=None, _b_range=(1.0, 1.5)), "prior"),
    "test_ood_pose": (3_000, P(pose_deg=6.0, pose_m=0.30), {}, "prior"),
    "test_ood_drift": (3_000, P(drift_max=0.06), {}, "prior"),
    "test_ood_site": (3_000, P(site_R=(265.0, 530.0)), {}, "prior"),
}

_SV = None


def _work(args):
    name, j, n = args
    global _SV
    if _SV is None:
        _SV = C.Survey(DX)
    total, prior, kw, mmode = SETS[name]
    kw = dict(kw)
    rng = np.random.default_rng(C.seed_for(f"ml:{name}:{j}"))
    if mmode == "prior":
        m = None
    elif mmode == "unit":
        m = unit_dirs(rng, n)
    else:
        m = np.tile(C.DIRS6[mmode], (n, 1))
    br = kw.pop("_b_range", None)
    if br is not None:
        kw["b"] = rng.uniform(br[0], br[1], n)
    o = C.simulate(_SV, rng, n, prior, m=m, **kw)
    o["z"] = o["z"].astype(np.float32)
    return o


def build(name, workers):
    total = SETS[name][0]
    jobs = [(name, j, min(CHUNK, total - j * CHUNK)) for j in range((total + CHUNK - 1) // CHUNK)]
    t0 = time.time()
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            parts = pool.map(_work, jobs)
    else:
        parts = [_work(a) for a in jobs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs(C.DATA, exist_ok=True)
    np.savez(os.path.join(C.DATA, f"{name}.npz"), **out)
    print(f"{name}: {total} surveys in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for name in a.sets.split(","):
        if os.path.exists(os.path.join(C.DATA, f"{name}.npz")):
            print(f"{name}: exists, skipped", flush=True); continue
        build(name, a.workers)


if __name__ == "__main__":
    main()
