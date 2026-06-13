"""
distreg.py — zero-inflated *distributional regression* for the precipitation
index.

Key idea (the structural redesign): instead of hurdling at the extreme threshold
u=1 and fitting a GPD to the ~300 exceedances, model the FULL zero-inflated
positive distribution of I using ALL wet days, and DERIVE P(I>u) and the tail
from it. Covariates drive the distribution parameters (linear predictors by
default; set deep=True for an MLP).

Positive-part distributions:
  * lognormal : log I ~ Normal(loc(x), scale)
  * gamma     : I ~ Gamma(shape, rate), mean = exp(loc(x))
(eGPD to be added if these confirm the structural hypothesis.)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = 2.0 ** 0.5


def _norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / SQRT2))


class _Head(nn.Module):
    """Linear or small-MLP map from covariates to a scalar predictor."""
    def __init__(self, d, deep=False):
        super().__init__()
        if deep:
            self.net = nn.Sequential(nn.Linear(d, 32), nn.Tanh(),
                                     nn.Linear(32, 1))
        else:
            self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ZIDistReg(nn.Module):
    def __init__(self, d, dist="lognormal", u=1.0, deep=False):
        super().__init__()
        assert dist in ("lognormal", "gamma", "egpd")
        self.dist, self.u = dist, u
        self.loc = _Head(d, deep)          # lognormal/gamma: loc; egpd: log-sigma
        self.p0 = _Head(d, deep)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.xi_raw = nn.Parameter(torch.tensor(-1.0))    # eGPD tail shape
        self.kappa_raw = nn.Parameter(torch.zeros(()))    # eGPD lower-tail shape

    def _xi(self):
        return 0.5 * torch.sigmoid(self.xi_raw)            # (0, 0.5), positive heavy tail

    def _kappa(self):
        return F.softplus(self.kappa_raw) + 0.05

    def _params(self, X):
        loc = self.loc(X)
        scale = F.softplus(self.log_scale) + 1e-3
        p0 = torch.sigmoid(self.p0(X)).clamp(1e-5, 1 - 1e-5)
        return p0, loc, scale

    def _Fpos(self, z, loc, scale):
        z = torch.clamp(z, min=1e-6)
        if self.dist == "lognormal":
            return _norm_cdf((torch.log(z) - loc) / scale)
        if self.dist == "gamma":
            k = 1.0 / (scale ** 2)
            rate = k / torch.exp(loc)
            return torch.special.gammainc(k * torch.ones_like(rate), rate * z)
        # eGPD: F(z) = H(z)^kappa, H = GPD CDF with scale=exp(loc), shape xi
        xi, kappa = self._xi(), self._kappa()
        sigma = torch.exp(loc)
        H = 1 - torch.clamp(1 + xi * z / sigma, min=1e-6) ** (-1.0 / xi)
        return torch.clamp(H, 1e-7, 1 - 1e-7) ** kappa

    def nll(self, X, I):
        p0, loc, scale = self._params(X)
        zero = I <= 1e-9
        Ipos = I.clamp_min(1e-6)
        if self.dist == "lognormal":
            logI = torch.log(Ipos)
            ll_pos = (torch.log1p(-p0) - logI - torch.log(scale)
                      - 0.5 * np.log(2 * np.pi) - 0.5 * ((logI - loc) / scale) ** 2)
        elif self.dist == "gamma":
            k = 1.0 / (scale ** 2)
            rate = k / torch.exp(loc)
            ll_pos = (torch.log1p(-p0) + k * torch.log(rate)
                      + (k - 1) * torch.log(Ipos) - rate * Ipos - torch.lgamma(k))
        else:  # eGPD: f(z) = kappa H^{kappa-1} h(z)
            xi, kappa = self._xi(), self._kappa()
            sigma = torch.exp(loc)
            base = torch.clamp(1 + xi * Ipos / sigma, min=1e-6)
            H = torch.clamp(1 - base ** (-1.0 / xi), 1e-7, 1 - 1e-7)
            log_h = -torch.log(sigma) + (-1.0 / xi - 1.0) * torch.log(base)
            ll_pos = (torch.log1p(-p0) + torch.log(kappa)
                      + (kappa - 1) * torch.log(H) + log_h)
        ll = torch.where(zero, torch.log(p0), ll_pos)
        return -ll.mean()

    def fit(self, X, I, epochs=400, lr=0.05, verbose=False):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            opt.zero_grad()
            loss = self.nll(X, I)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            opt.step()
            if verbose and e % (epochs // 5) == 0:
                print(f"    distreg[{self.dist}] ep{e} nll={loss.item():.4f}")
        return self

    @torch.no_grad()
    def p_exceed(self, X, u=None):
        u = self.u if u is None else u
        p0, loc, scale = self._params(X)
        Fu = self._Fpos(torch.tensor(float(u)), loc, scale)
        return ((1 - p0) * (1 - Fu)).numpy()

    @torch.no_grad()
    def cdf_grid(self, X, z):
        """Predictive CDF on grid z (np, nz) -> (n, nz)."""
        p0, loc, scale = self._params(X)
        zt = torch.tensor(z, dtype=torch.float32)
        Fpos = self._Fpos(zt[None, :], loc[:, None], scale)
        return (p0[:, None] + (1 - p0[:, None]) * Fpos).numpy()


class UnifiedHurdle(nn.Module):
    """A single hurdle predictive distribution fit by ONE joint likelihood.

    F(z | x) = (1 - pi(x))                         for z <= u
             = (1 - pi(x)) + pi(x) H(z-u; s(x), xi) for z >  u

    with occurrence  pi(x)    = sigmoid(a . x + a0)
    and  intensity   sigma(x) = softplus(b . x + b0),  xi constant (covariate-
    free tail index, for stability on ~1-5% exceedances). H is the GPD CDF of
    the excess. The occurrence (Bernoulli) and intensity (GPD) terms are the two
    additive parts of ONE log-likelihood; the whole F is scored end-to-end by a
    single threshold-weighted CRPS, which decomposes back into an occurrence and
    an intensity contribution. (The unstructured likelihood factorises, so this
    joint fit equals fitting the parts separately -- the components are orthogonal
    pieces of one distribution, not two disconnected analyses; coupling them
    requires shared latent structure, tested separately via the deep-GP hurdle.)
    """
    def __init__(self, d, u=1.0):
        super().__init__()
        self.u = u
        self.a = nn.Linear(d, 1)        # occurrence linear predictor
        self.b = nn.Linear(d, 1)        # log-scale linear predictor
        self.xi_raw = nn.Parameter(torch.tensor(-1.0))

    def _xi(self):
        return 0.5 * torch.sigmoid(self.xi_raw)        # (0, 0.5), heavy positive tail

    def params(self, X):
        pi = torch.sigmoid(self.a(X).squeeze(-1)).clamp(1e-5, 1 - 1e-5)
        sigma = F.softplus(self.b(X).squeeze(-1)) + 1e-3
        return pi, sigma, self._xi()

    def nll(self, X, O, excess):
        pi, sigma, xi = self.params(X)
        occ = -(O * torch.log(pi) + (1 - O) * torch.log1p(-pi))
        e = excess.clamp_min(0.0)
        base = torch.clamp(1 + xi * e / sigma, min=1e-6)
        gpd_ll = -torch.log(sigma) + (-1.0 / xi - 1.0) * torch.log(base)
        intens = torch.where(O > 0.5, -gpd_ll, torch.zeros_like(gpd_ll))
        return (occ + intens).mean()

    def fit(self, X, O, excess, epochs=400, lr=0.05, verbose=False):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            opt.zero_grad()
            loss = self.nll(X, O, excess)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            opt.step()
            if verbose and e % max(1, epochs // 5) == 0:
                print(f"    unified-hurdle ep{e} nll={loss.item():.4f}")
        return self

    @torch.no_grad()
    def predict(self, X):
        pi, sigma, xi = self.params(X)
        return pi.numpy(), sigma.numpy(), float(xi)
