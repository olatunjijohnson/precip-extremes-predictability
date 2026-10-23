"""
tcdgp.py — Tail-Calibrated (deep) Gaussian-process EVT hurdle model.

A self-contained PyTorch implementation of the model in the paper
(Paper/paper/main.tex, Section 4). No gpytorch dependency.

Components
----------
* FeatureExtractor : optional MLP feature map  -> the "deep kernel".
* RBFKernel        : ARD squared-exponential kernel on the feature map.
* CoregSVGP        : Q base sparse variational GPs (shared inducing inputs Z),
                     mixed by W (2 x Q) into two correlated latent functions
                     (g_pi, g_sigma).  B = W W^T is the coregionalisation matrix;
                     rho = B_01 / sqrt(B_00 B_11) is the borrowing-strength
                     parameter.  Setting coupled=False fixes W diagonal -> the
                     independent hurdle (clean nested ablation).
* HurdleGPD        : occurrence (Bernoulli) + intensity (Generalised Pareto)
                     likelihood; constant shape xi.
* TCDGP            : ties it together, exposes elbo(), fit(), predict_params().

The maths (whitened SVGP, ELBO) is documented inline so it can lift into the
methods appendix.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

JIT = 1e-5  # kernel jitter


# --------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """MLP feature map (deep kernel).  deep=False -> identity (plain GP)."""
    def __init__(self, d_in, d_out=4, hidden=(32, 16), deep=True):
        super().__init__()
        if not deep:
            self.net, self.d_out = None, d_in
        else:
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            layers += [nn.Linear(prev, d_out)]
            self.net, self.d_out = nn.Sequential(*layers), d_out

    def forward(self, x):
        return x if self.net is None else self.net(x)


# --------------------------------------------------------------------------
class RBFKernel(nn.Module):
    """ARD squared-exponential kernel."""
    def __init__(self, d):
        super().__init__()
        self.log_ls = nn.Parameter(torch.zeros(d))   # log lengthscales
        self.log_os = nn.Parameter(torch.zeros(()))  # log outputscale

    def forward(self, X1, X2):
        ls = torch.exp(self.log_ls)
        X1s, X2s = X1 / ls, X2 / ls
        d2 = (X1s**2).sum(1, keepdim=True) - 2 * X1s @ X2s.t() + (X2s**2).sum(1)[None, :]
        return torch.exp(self.log_os) * torch.exp(-0.5 * d2.clamp_min(0))

    def diag(self, X):
        return torch.exp(self.log_os) * torch.ones(X.shape[0], device=X.device)


# --------------------------------------------------------------------------
class CoregSVGP(nn.Module):
    """Q whitened sparse-variational base GPs mixed into two latent outputs."""
    def __init__(self, d_feat, M=64, Q=2, coupled=True):
        super().__init__()
        self.Q, self.coupled = Q, coupled
        self.kernel = RBFKernel(d_feat)
        self.Z = nn.Parameter(torch.randn(M, d_feat) * 0.5)      # inducing inputs
        self.m = nn.Parameter(torch.zeros(Q, M))                 # variational mean (whitened)
        # variational covariance Cholesky factors (lower-tri), one per base GP
        self.L_raw = nn.Parameter(torch.stack([0.1 * torch.eye(M) for _ in range(Q)]))
        # mixing matrix W (2 x Q): B = W W^T
        if coupled:
            self.W = nn.Parameter(torch.eye(2, Q) + 0.01 * torch.randn(2, Q))
        else:                                                    # independent hurdle
            assert Q == 2
            self.w_diag = nn.Parameter(torch.ones(2))

    def _W(self):
        if self.coupled:
            return self.W
        return torch.diag(self.w_diag)                            # 2 x 2, off-diag = 0

    def _Schol(self):
        L = torch.tril(self.L_raw)
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-4
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return L                                                 # (Q, M, M)

    def base_marginals(self, Phi):
        """Predictive mean/var of each base GP at features Phi.  Returns (n,Q),(n,Q)."""
        M = self.Z.shape[0]
        Kmm = self.kernel(self.Z, self.Z) + JIT * torch.eye(M, device=Phi.device)
        Lmm = torch.linalg.cholesky(Kmm)
        Knm = self.kernel(Phi, self.Z)                           # n x M
        # A = Knm @ Lmm^{-T} ; via X = Lmm^{-1} Knm^T -> A = X^T
        X = torch.linalg.solve_triangular(Lmm, Knm.t(), upper=False)  # M x n
        A = X.t()                                                # n x M
        kxx = self.kernel.diag(Phi)                              # n
        Lsq = self._Schol()                                      # Q,M,M
        means, varis = [], []
        for q in range(self.Q):
            mean_q = A @ self.m[q]                               # n
            AS = A @ Lsq[q]                                      # n x M  (A L_S)
            var_q = kxx - (A * A).sum(1) + (AS * AS).sum(1)      # n
            means.append(mean_q)
            varis.append(var_q.clamp_min(1e-6))
        return torch.stack(means, 1), torch.stack(varis, 1)      # n,Q

    def kl(self):
        """Sum of KL(q(v_q)||N(0,I)) over base GPs (whitened prior)."""
        M = self.Z.shape[0]
        Lsq = self._Schol()
        tot = 0.0
        for q in range(self.Q):
            S_diag = (Lsq[q]**2).sum()                           # tr(S)
            logdet = 2 * torch.log(torch.diagonal(Lsq[q])).sum()
            tot = tot + 0.5 * (S_diag + (self.m[q]**2).sum() - M - logdet)
        return tot

    def sample_latent(self, Phi, n_samples=1):
        """Correlated samples of (g_pi, g_sigma).  Returns (S, n, 2)."""
        means, varis = self.base_marginals(Phi)                  # n,Q
        eps = torch.randn(n_samples, *means.shape, device=Phi.device)
        f = means.unsqueeze(0) + varis.sqrt().unsqueeze(0) * eps  # S,n,Q
        g = f @ self._W().t()                                    # S,n,2  (mix)
        return g

    def B(self):
        W = self._W()
        return W @ W.t()

    def rho(self):
        B = self.B()
        return (B[0, 1] / (B[0, 0] * B[1, 1]).sqrt()).item()


# --------------------------------------------------------------------------
class HurdleGPD(nn.Module):
    """Occurrence (Bernoulli) + intensity (GPD) likelihood, constant shape xi."""
    def __init__(self, u=1.0, xi_init=0.1):
        super().__init__()
        self.u = u
        self.xi_raw = nn.Parameter(torch.tensor(math.atanh(xi_init / 0.5)))

    def xi(self):
        return 0.5 * torch.tanh(self.xi_raw)                     # in (-0.5, 0.5)

    @staticmethod
    def _gpd_logpdf(y, sigma, xi):
        z = 1 + xi * y / sigma
        ll_gen = -torch.log(sigma) - (1.0 / xi + 1.0) * torch.log(z.clamp_min(1e-6))
        ll_exp = -torch.log(sigma) - y / sigma
        ll = torch.where(xi.abs() < 1e-4, ll_exp, ll_gen)
        return torch.where(z > 0, ll, torch.full_like(ll, -1e4))  # outside support

    def loglik(self, g, O, excess):
        """g:(...,n,2)  O:(n,) in {0,1}  excess:(n,) = I-u (valid where O=1)."""
        pi = torch.sigmoid(g[..., 0]).clamp(1e-6, 1 - 1e-6)
        sigma = F.softplus(g[..., 1]) + 1e-4
        occ = O * torch.log(pi) + (1 - O) * torch.log(1 - pi)
        gpd = self._gpd_logpdf(excess.clamp_min(0.0), sigma, self.xi())
        return occ + O * gpd                                     # (...,n)


# --------------------------------------------------------------------------
class TCDGP(nn.Module):
    def __init__(self, d_in, u=1.0, M=64, Q=2, coupled=True, deep=True, feat_dim=4):
        super().__init__()
        self.fe = FeatureExtractor(d_in, feat_dim, deep=deep)
        self.gp = CoregSVGP(self.fe.d_out, M=M, Q=Q, coupled=coupled)
        self.lik = HurdleGPD(u=u)
        self.u = u

    def elbo(self, X, O, excess, n_data=None, n_samples=8):
        Phi = self.fe(X)
        g = self.gp.sample_latent(Phi, n_samples)                # S,n,2
        ll = self.lik.loglik(g, O, excess).mean(0)               # n  (avg over samples)
        scale = (n_data / X.shape[0]) if n_data else 1.0         # minibatch correction
        return scale * ll.sum() - self.gp.kl()

    def fit(self, X, O, excess, epochs=400, lr=0.01, batch=None, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = X.shape[0]
        for ep in range(epochs):
            opt.zero_grad()
            if batch and batch < n:
                idx = torch.randperm(n)[:batch]
                loss = -self.elbo(X[idx], O[idx], excess[idx], n_data=n)
            else:
                loss = -self.elbo(X, O, excess)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            opt.step()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  -elbo={loss.item():.1f}  "
                      f"xi={self.lik.xi().item():.3f}  rho={self.gp.rho():+.3f}")
        return self

    @torch.no_grad()
    def predict_params(self, X, n_samples=200):
        """MC-averaged predictive pi, sigma; plus xi.  Returns dict of (n,) tensors."""
        Phi = self.fe(X)
        g = self.gp.sample_latent(Phi, n_samples)                # S,n,2
        pi = torch.sigmoid(g[..., 0]).mean(0)
        sigma = (F.softplus(g[..., 1]) + 1e-4).mean(0)
        return {"pi": pi, "sigma": sigma, "xi": self.lik.xi().detach()}

    @torch.no_grad()
    def predictive_cdf(self, z, params):
        """F(z) = (1-pi) + pi H(z-u) for z>u ; (1-pi)*(z/u) crude bulk for z<=u."""
        pi, sigma, xi = params["pi"], params["sigma"], params["xi"]
        y = (z - self.u).clamp_min(0.0)
        H = 1 - (1 + xi * y / sigma).clamp_min(1e-6) ** (-1.0 / xi) if abs(xi) > 1e-4 \
            else 1 - torch.exp(-y / sigma)
        tail = (1 - pi) + pi * H
        bulk = (1 - pi) * (z / self.u).clamp(0, 1)
        return torch.where(torch.as_tensor(z) > self.u, tail, bulk)

    @torch.no_grad()
    def tail_quantile(self, level, params):
        """Smallest z with F(z) >= level, using the GPD tail (z>u).  Returns (n,)."""
        pi, sigma, xi = params["pi"], params["sigma"], params["xi"]
        # need level > 1-pi for an exceedance; else return u
        frac = ((1 - level) / pi).clamp(1e-8, 1.0)               # (1-tau)/pi
        if abs(xi) > 1e-4:
            z = self.u + sigma / xi * (frac ** (-xi) - 1)
        else:
            z = self.u - sigma * torch.log(frac)
        return torch.where(torch.as_tensor(level) > (1 - pi), z, torch.full_like(z, self.u))
