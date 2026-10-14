"""A small multilayer perceptron with a full-covariance Gaussian output, written in numpy (no deep-learning
framework, so the thesis environment needs nothing new). Backpropagation is by hand and is checked against
finite differences by `gradient_check`.

Network:  x (standardized features) -> [Linear -> SiLU] x L -> Linear -> (delta_mu, eta, l)
          mu = W_skip x + b_skip + delta_mu           (trainable linear skip, initialized at least squares)
          precision  Lambda = L L^T,  L = [[e^eta0, 0, 0], [l0, e^eta1, 0], [l1, l2, e^eta2]]
Loss:     Gaussian negative log-likelihood per sample (constant dropped)
          NLL = 0.5 |L^T (y - mu)|^2 - sum_i eta_i
The network outputs mu and Sigma = Lambda^{-1}; the mean is the point estimate.
"""
import numpy as np


def silu(a):
    s = 1.0 / (1.0 + np.exp(-a))
    return a * s, s


class GaussMLP:
    def __init__(self, n_in, hidden=(256, 256, 256), seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        sizes = [n_in, *hidden, 9]
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / a) if i < len(sizes) - 2 else 1e-3
            self.W.append((rng.normal(0, scale, (a, c))).astype(dtype))
            self.b.append(np.zeros(c, dtype))
        self.Ws = np.zeros((n_in, 3), dtype); self.bs = np.zeros(3, dtype)
        self.b[-1][3:6] = 0.0

    # parameters as a flat list for the optimizer
    def params(self):
        return self.W + self.b + [self.Ws, self.bs]

    def forward(self, x, cache=False):
        h = x; acts = [x]; pre = []
        for i in range(len(self.W) - 1):
            a = h @ self.W[i] + self.b[i]
            h, s = silu(a)
            pre.append((a, s)); acts.append(h)
        o = h @ self.W[-1] + self.b[-1]
        mu = x @ self.Ws + self.bs + o[:, :3]
        eta = o[:, 3:6]; l = o[:, 6:9]
        if cache:
            self._cache = (acts, pre)
        return mu, eta, l

    @staticmethod
    def chol(eta, l):
        n = len(eta)
        L = np.zeros((n, 3, 3), eta.dtype)
        e = np.exp(np.clip(eta, -20, 20))
        L[:, 0, 0] = e[:, 0]; L[:, 1, 1] = e[:, 1]; L[:, 2, 2] = e[:, 2]
        L[:, 1, 0] = l[:, 0]; L[:, 2, 0] = l[:, 1]; L[:, 2, 1] = l[:, 2]
        return L, e

    def loss_and_grad(self, x, y, mse_only=False):
        mu, eta, l = self.forward(x, cache=True)
        r = y - mu
        if mse_only:      # warm-up: fit the mean alone
            loss = 0.5 * np.mean(np.sum(r * r, 1))
            g_mu = -r / len(x)
            g_o = np.zeros((len(x), 9), x.dtype); g_o[:, :3] = g_mu
            return loss, self._backward(x, g_o, g_mu)
        L, e = self.chol(eta, l)
        s = np.einsum("nij,ni->nj", L, r)                         # s = L^T r
        loss = np.mean(0.5 * np.sum(s * s, 1) - np.sum(eta, 1))
        n = len(x)
        g_mu = -np.einsum("nij,nj->ni", L, s) / n                  # dNLL/dmu = -L s
        G = r[:, :, None] * s[:, None, :]                          # dNLL/dL_ij = r_i s_j
        g_eta = (np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2]], 1) * e - 1.0) / n
        g_l = np.stack([G[:, 1, 0], G[:, 2, 0], G[:, 2, 1]], 1) / n
        g_o = np.concatenate([g_mu, g_eta, g_l], 1).astype(x.dtype)
        return loss, self._backward(x, g_o, g_mu)

    def _backward(self, x, g_o, g_mu):
        acts, pre = self._cache
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        gW[-1] = acts[-1].T @ g_o; gb[-1] = g_o.sum(0)
        g = g_o @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            a, s = pre[i]
            g = g * (s * (1.0 + a * (1.0 - s)))                     # d silu / da
            gW[i] = acts[i].T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W[i].T
        gWs = x.T @ g_mu; gbs = g_mu.sum(0)
        return gW + gb + [gWs, gbs]

    def predict(self, x, batch=8192):
        mus, covs = [], []
        for i in range(0, len(x), batch):
            mu, eta, l = self.forward(x[i:i + batch])
            L, _ = self.chol(eta.astype(np.float64), l.astype(np.float64))
            P = L @ np.transpose(L, (0, 2, 1))
            mus.append(mu.astype(np.float64)); covs.append(np.linalg.inv(P))
        return np.concatenate(mus), np.concatenate(covs)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)


def gradient_check(seed=0, n_in=7, n=5):
    """Max relative error between the analytic gradient and central differences (float64)."""
    rng = np.random.default_rng(seed)
    net = GaussMLP(n_in, hidden=(6, 5), seed=seed, dtype=np.float64)
    for p in net.params():
        p += rng.normal(0, 0.3, p.shape)
    x = rng.normal(size=(n, n_in)); y = rng.normal(size=(n, 3))
    _, grads = net.loss_and_grad(x, y)
    worst = 0.0
    for p, g in zip(net.params(), grads):
        flat = p.reshape(-1); gf = g.reshape(-1)
        for j in rng.choice(flat.size, size=min(flat.size, 6), replace=False):
            old = flat[j]; h = 1e-6
            flat[j] = old + h; lp, _ = net.loss_and_grad(x, y)
            flat[j] = old - h; lm, _ = net.loss_and_grad(x, y)
            flat[j] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - gf[j]) / max(1e-8, abs(num) + abs(gf[j])))
    return worst


if __name__ == "__main__":
    print("gradient check, max relative error:", gradient_check())
