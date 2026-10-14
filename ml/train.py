"""Train one Gaussian-output network on a simulated training set.

Features x = [Qr^T d, ln(residual energy / dof)], standardized with training-set statistics. The linear skip of
the network is initialized at the 11-column least-squares estimator expressed in these features, so the network
starts as the thesis estimator and learns a correction. Three epochs fit the mean alone (squared error); the rest
minimize the Gaussian negative log-likelihood. Learning rate: cosine decay from 1e-3 to 1e-5. The weights with the
lowest validation NLL are kept.

Run:  python ml/train.py --set full --seed 0 [--epochs 40]
Out:  ml/models/<set>_s<seed>.npz, ml/results/train_<set>_s<seed>.json
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from nn import GaussMLP, Adam

MODELS = os.path.join(C.HERE, "models")


def features(o, dof, n_inv=8):
    z = np.asarray(o["z"], np.float64)[:, n_inv:]
    return np.column_stack([z, np.log(np.asarray(o["res"], float) / dof)])


def load(name):
    return dict(np.load(os.path.join(C.DATA, f"{name}.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", default="256,256,256"); ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    sv = C.Survey(0.5)
    dof = sv.nd - sv.k
    tr, va = load(f"train_{a.set}"), load(f"val_{a.set}")
    Xtr, Xva = features(tr, dof), features(va, dof)
    Ytr, Yva = tr["M"].astype(np.float32), va["M"].astype(np.float32)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0)
    Xtr = ((Xtr - mu_x) / sd_x).astype(np.float32); Xva = ((Xva - mu_x) / sd_x).astype(np.float32)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    net = GaussMLP(Xtr.shape[1], hidden=hidden, seed=C.seed_for(f"ml:init:{a.set}:{a.seed}") % 2 ** 31)
    # linear skip = LS11 in standardized features (weight on the residual feature 0)
    Kinv = sv.est_z["LS11"][:, sv.n_inv:]                      # (3, k - 8)
    Ws = np.zeros((Xtr.shape[1], 3)); Ws[:-1] = (Kinv * sd_x[None, :-1]).T
    net.Ws[:] = Ws.astype(np.float32); net.bs[:] = (Kinv @ mu_x[:-1]).astype(np.float32)
    opt = Adam(net.params(), lr=1e-3)
    rng = np.random.default_rng(C.seed_for(f"ml:shuffle:{a.set}:{a.seed}"))
    nb = len(Xtr) // a.batch
    total = a.epochs * nb
    hist = []; best = (np.inf, None, -1); t0 = time.time(); step = 0

    def val_nll():
        mu, eta, l = net.forward(Xva)
        L, _ = net.chol(eta.astype(np.float64), l.astype(np.float64))
        r = Yva - mu
        s = np.einsum("nij,ni->nj", L, r)
        nll = np.mean(0.5 * np.sum(s * s, 1) - np.sum(eta, 1)) + 1.5 * np.log(2 * np.pi)
        return float(nll), float(np.mean(np.sum(r * r, 1)))

    for ep in range(a.epochs):
        perm = rng.permutation(len(Xtr)); run = 0.0
        for bi in range(nb):
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * step / total))
            loss, g = net.loss_and_grad(Xtr[idx], Ytr[idx], mse_only=ep < a.warmup)
            opt.step(net.params(), g, lr=lr); run += loss; step += 1
        nll, mse = val_nll()
        hist.append({"epoch": ep, "train_loss": float(run / nb), "val_nll": nll, "val_mse": mse, "t_s": time.time() - t0})
        print(f"[{a.set} s{a.seed}] epoch {ep:2d} train {run/nb:.4f} val NLL {nll:.4f} MSE {mse:.5f} ({time.time()-t0:.0f} s)", flush=True)
        if ep >= a.warmup and nll < best[0]:
            best = (nll, [p.copy() for p in net.params()], ep)
    for p, b in zip(net.params(), best[1]):
        p[:] = b
    os.makedirs(MODELS, exist_ok=True)
    np.savez(os.path.join(MODELS, f"{a.set}_s{a.seed}.npz"), mu_x=mu_x, sd_x=sd_x, hidden=np.array(hidden),
             **{f"p{i}": p for i, p in enumerate(net.params())})
    json.dump({"set": a.set, "seed": a.seed, "best_epoch": best[2], "best_val_nll": best[0], "history": hist,
               "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_features": int(Xtr.shape[1]), "hidden": hidden,
               "batch": a.batch, "epochs": a.epochs, "warmup_epochs": a.warmup},
              open(os.path.join(C.RESULTS, f"train_{a.set}_s{a.seed}.json"), "w"), indent=1)


def load_net(path):
    f = np.load(path)
    hidden = tuple(int(h) for h in f["hidden"])
    n_in = f["p0"].shape[0]
    net = GaussMLP(n_in, hidden=hidden)
    for i, p in enumerate(net.params()):
        p[...] = f[f"p{i}"]
    return net, f["mu_x"], f["sd_x"]


if __name__ == "__main__":
    main()
