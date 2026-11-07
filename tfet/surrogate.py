"""Neural surrogates for log10(I_D): plain MLP, physics-informed MLP and deep ensembles.

Physics-informed here means three things the paper's random forest does not have:
  1. target is log10(I_D), so the subthreshold region (I_OFF, SS) is learned as well as I_ON
  2. optional physics features: tunnelling length lambda and tunnelling area A_tun
  3. a monotonicity penalty  relu(-d logI / dV_G)  on random collocation points,
     so every predicted transfer curve rises with V_G and SS/g_m are well defined
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .physics import W_SHEET, tunnelling_length

torch.set_num_threads(max(1, torch.get_num_threads()))


def add_physics_features(X: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Append lambda = sqrt(eps t_ox t_n / eps_ox) and log A_tun = log(L_ov * 2 (W + t_n))."""
    c = {n: X[:, i] for i, n in enumerate(cols)}
    lam = tunnelling_length(c["t_ox"], c["t_n"])
    log_area = np.log(c["L_ov"] * 2 * (W_SHEET + c["t_n"]))
    return np.column_stack([X, lam, log_area]), cols + ["lambda", "log_A_tun"]


class Standardizer:
    def fit(self, a):
        self.mu, self.sd = a.mean(0), a.std(0) + 1e-12
        return self

    def transform(self, a):
        return (a - self.mu) / self.sd

    def inverse(self, a):
        return a * self.sd + self.mu


class MLP(nn.Module):
    def __init__(self, n_in, width=128, depth=4):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogCurrentSurrogate:
    """Point-wise surrogate  (V_G, device params, ...) -> log10 I_D.

    `cols` names the input columns; it must contain 'V_G'. With physics=True the
    lambda / A_tun features are appended and the monotonicity penalty is used.
    """

    def __init__(self, cols, physics=True, width=128, depth=4, mono_weight=1.0, seed=0,
                 n_colloc=1024):
        self.cols, self.physics = list(cols), physics
        self.width, self.depth, self.mono_weight, self.seed = width, depth, mono_weight, seed
        self.n_colloc = n_colloc      # random points per step for the monotonicity penalty

    def _features(self, X):
        return add_physics_features(X, self.cols)[0] if self.physics else X

    def fit(self, X, I_D, epochs=300, batch=4096, lr=3e-3, verbose=False):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        F = self._features(X)
        y = np.log10(I_D)
        self.xs, self.ys = Standardizer().fit(F), Standardizer().fit(y)
        Xt = torch.tensor(self.xs.transform(F), dtype=torch.float32)
        yt = torch.tensor(self.ys.transform(y), dtype=torch.float32)
        self.lo, self.hi = X.min(0), X.max(0)     # box for collocation points
        self.vg = self.cols.index("V_G")

        self.model = MLP(Xt.shape[1], self.width, self.depth)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        steps = epochs * int(np.ceil(len(Xt) / batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        self.history = []
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch):
                idx = perm[i:i + batch]
                loss = nn.functional.mse_loss(self.model(Xt[idx]), yt[idx])
                if self.physics and self.mono_weight > 0:
                    loss = loss + self.mono_weight * self._mono_penalty(rng, self.n_colloc)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            self.history.append(loss.item())
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"epoch {ep:4d}  loss {loss.item():.5f}")
        return self

    def _mono_penalty(self, rng, n):
        """Penalise negative slope d logI / dV_G at random points inside the training box."""
        Xc = rng.uniform(self.lo, self.hi, size=(n, len(self.cols)))
        xc = torch.tensor(self.xs.transform(self._features(Xc)), dtype=torch.float32,
                          requires_grad=True)
        grad = torch.autograd.grad(self.model(xc).sum(), xc, create_graph=True)[0][:, self.vg]
        return torch.relu(-grad).pow(2).mean()

    @torch.no_grad()
    def predict_log(self, X):
        F = torch.tensor(self.xs.transform(self._features(X)), dtype=torch.float32)
        return self.ys.inverse(self.model(F).numpy())

    def predict(self, X):
        return 10 ** self.predict_log(X)


def train_run(X, I_D, X_test, cols, physics, mono_weight=1.0, seed=0, steps=5000, batch=4096,
              threads=2):
    """Train one surrogate for a fixed number of optimiser steps; return (model, preds, seconds).

    A fixed step budget keeps small training sets from being under-trained. It is a plain
    module-level function so that joblib can run several trainings side by side.
    """
    import time
    torch.set_num_threads(threads)
    epochs = int(np.ceil(steps / np.ceil(len(X) / batch)))
    t0 = time.perf_counter()
    model = LogCurrentSurrogate(cols, physics=physics, mono_weight=mono_weight, seed=seed)
    model.fit(X, I_D, epochs=epochs, batch=batch)
    return model, model.predict(X_test), time.perf_counter() - t0


class DeepEnsemble:
    """M independently initialised surrogates; spread between members = epistemic uncertainty."""

    def __init__(self, cols, n_members=5, **kw):
        self.members = [LogCurrentSurrogate(cols, seed=s, **kw) for s in range(n_members)]

    def fit(self, X, I_D, **kw):
        for m in self.members:
            m.fit(X, I_D, **kw)
        return self

    def predict_log(self, X):
        """Return mean and standard deviation of log10 I_D over the ensemble."""
        P = np.stack([m.predict_log(X) for m in self.members])
        return P.mean(0), P.std(0)
