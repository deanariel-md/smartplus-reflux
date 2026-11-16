"""Poisson GLMM with a normal random intercept per cluster, in plain numpy/scipy.

statsmodels has no frequentist Poisson GLMM, so this fits
    y_ij ~ Poisson(exp(x_ij'b + offset_ij + u_j)),  u_j ~ N(0, s^2)
by maximum likelihood with adaptive Gauss-Hermite quadrature (AGHQ).

For a random intercept the cluster integral depends on the data only through
S_y = sum_i y_ij and S_mu = sum_i exp(x_ij'b + offset_ij), so each likelihood
evaluation costs one pass over the rows. Gradients are analytic; the Hessian
is a central difference of the gradient. Standard errors are model-based and
cluster-robust (sandwich over clusters), the latter being the default for
reporting because hourly and daily reflux counts are overdispersed.

    res = fit(y, X, offset, groups, names)
    res.table()   -> DataFrame with estimates, robust SE, rate ratios and 95% CI
"""
import numpy as np
import pandas as pd
from scipy import optimize, special

_Z, _W = np.polynomial.hermite.hermgauss(25)   # nodes/weights for int e^{-z^2} f(z) dz


class Result:
    def __init__(self, names, params, cov_model, cov_robust, sigma, loglik, n_obs, n_groups, converged):
        self.names, self.params = names, params
        self.cov_model, self.cov_robust = cov_model, cov_robust
        self.sigma, self.loglik, self.n_obs, self.n_groups, self.converged = sigma, loglik, n_obs, n_groups, converged

    def table(self, robust=True, exponentiate=True):
        p = len(self.names)
        se = np.sqrt(np.diag(self.cov_robust if robust else self.cov_model))[:p]
        b = self.params[:p]
        z = b / se
        out = pd.DataFrame({"term": self.names, "estimate": b, "se": se,
                            "p": 2 * (1 - special.ndtr(np.abs(z)))})
        if exponentiate:
            out["RR"] = np.exp(b)
            out["RR_lo"] = np.exp(b - 1.959964 * se)
            out["RR_hi"] = np.exp(b + 1.959964 * se)
        return out

    def summary(self):
        t = self.table()
        head = (f"Poisson GLMM, random intercept (AGHQ {len(_Z)} nodes): n={self.n_obs}, clusters={self.n_groups}, "
                f"sigma={self.sigma:.3f}, loglik={self.loglik:.1f}, converged={self.converged}; robust SE")
        return head + "\n" + t.round(4).to_string(index=False)


def _cluster_terms(theta, y, X, off, g, G, Sy, lgam_sum):
    """Per-cluster log-likelihood and score contributions."""
    p = X.shape[1]
    beta, logs = theta[:p], theta[p]
    s2 = np.exp(2 * logs)
    eta = X @ beta + off
    mu = np.exp(eta)
    Smu = np.bincount(g, weights=mu, minlength=G)
    Syeta = np.bincount(g, weights=y * eta, minlength=G)

    # Mode of h(u) = Sy*u - Smu*e^u - u^2/(2 s2) by Newton (concave in u).
    u = np.log((Sy + 0.5) / (Smu + 0.5 * np.exp(0))) * s2 / (s2 + 1.0)
    for _ in range(50):
        d1 = Sy - Smu * np.exp(u) - u / s2
        d2 = -Smu * np.exp(u) - 1.0 / s2
        step = d1 / d2
        u = u - step
        if np.max(np.abs(step)) < 1e-10:
            break
    curv = Smu * np.exp(u) + 1.0 / s2
    scale = np.sqrt(2.0 / curv)
    nodes = u[:, None] + scale[:, None] * _Z[None, :]                  # G x K
    h = Sy[:, None] * nodes - Smu[:, None] * np.exp(nodes) - nodes ** 2 / (2 * s2)
    logw = np.log(_W)[None, :] + _Z[None, :] ** 2 + h
    m = logw.max(axis=1, keepdims=True)
    wts = np.exp(logw - m)
    I = wts.sum(axis=1)
    logI = np.log(I) + m[:, 0] + np.log(scale)
    post = wts / I[:, None]                                            # normalized posterior weights
    E_eu = (post * np.exp(nodes)).sum(axis=1)
    E_u2 = (post * nodes ** 2).sum(axis=1)

    ll_g = Syeta - lgam_sum - 0.5 * np.log(2 * np.pi * s2) + logI
    # Scores: d/dbeta = sum_i y x - E[e^u] sum_i mu x ; d/dlogs = -1 + E[u^2]/s2
    resid = y - E_eu[g] * mu
    score_b = np.zeros((G, p))
    for k in range(p):
        score_b[:, k] = np.bincount(g, weights=resid * X[:, k], minlength=G)
    score_s = -1.0 + E_u2 / s2
    return ll_g, np.column_stack([score_b, score_s])


def fit(y, X, offset, groups, names=None, maxiter=500):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    g, G = codes, len(uniq)
    Sy = np.bincount(g, weights=y, minlength=G)
    lgam_sum = np.bincount(g, weights=special.gammaln(y + 1), minlength=G)
    p = X.shape[1]
    names = list(names) if names is not None else [f"x{k}" for k in range(p)]

    # Start from a pooled Poisson GLM (Newton steps), random-intercept SD 0.5.
    b0 = np.zeros(p)
    if np.allclose(X[:, 0], 1):
        b0[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(25):
        mu = np.exp(X @ b0 + off)
        step = np.linalg.solve(X.T @ (X * mu[:, None]) + 1e-8 * np.eye(p), X.T @ (y - mu))
        b0 = b0 + step
        if np.max(np.abs(step)) < 1e-8:
            break
    theta0 = np.append(b0, np.log(0.5))

    def nll(th):
        ll, sc = _cluster_terms(th, y, X, off, g, G, Sy, lgam_sum)
        return -ll.sum(), -sc.sum(axis=0)

    r = optimize.minimize(nll, theta0, jac=True, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6})
    th = r.x
    # BFGS often stops on "precision loss" at the optimum; judge convergence by the gradient.
    converged = bool(r.success or np.max(np.abs(nll(th)[1])) < 1e-3)
    # Hessian by central differences of the analytic gradient.
    k = len(th)
    H = np.zeros((k, k))
    for j in range(k):
        eps = 1e-5 * max(1.0, abs(th[j]))
        e = np.zeros(k)
        e[j] = eps
        H[:, j] = (nll(th + e)[1] - nll(th - e)[1]) / (2 * eps)
    H = (H + H.T) / 2
    bread = np.linalg.inv(H)
    _, sc = _cluster_terms(th, y, X, off, g, G, Sy, lgam_sum)
    meat = sc.T @ sc * G / max(G - 1, 1)
    robust = bread @ meat @ bread
    return Result(names, th, bread, robust, float(np.exp(th[p])), -r.fun, len(y), G, converged)


def design(df, formula_terms, intercept=True):
    """Small helper: build X from numeric columns and categorical dummies.

    formula_terms: list of column names; names ending with ':C' are treated as
    categorical with the first sorted level as reference."""
    cols, names = [], []
    if intercept:
        cols.append(np.ones(len(df)))
        names.append("(Intercept)")
    for t in formula_terms:
        if t.endswith(":C"):
            c = t[:-2]
            levels = sorted(df[c].dropna().unique())
            for lv in levels[1:]:
                cols.append((df[c] == lv).astype(float).to_numpy())
                names.append(f"{c}[{lv}]")
        else:
            cols.append(df[t].astype(float).to_numpy())
            names.append(t)
    return np.column_stack(cols), names


if __name__ == "__main__":
    # Self-test: recover known parameters from simulated clustered counts with exposure offsets.
    rng = np.random.default_rng(7)
    G, n = 45, 150
    grp = np.repeat(np.arange(G), n)
    u = rng.normal(0, 0.8, G)[grp]
    x1 = rng.normal(size=G * n)
    x2 = rng.integers(0, 3, G * n)
    expo = rng.uniform(0.3, 1.0, G * n)
    eta = 0.9 + 0.25 * x1 - 0.4 * (x2 == 1) + 0.3 * (x2 == 2) + np.log(expo) + u
    yy = rng.poisson(np.exp(eta))
    df = pd.DataFrame({"x1": x1, "x2": x2})
    Xs, nm = design(df, ["x1", "x2:C"])
    res = fit(yy, Xs, np.log(expo), grp, nm)
    print(res.summary())
    uj = u[::n]
    print(f"truth: x1=0.25, x2[1]=-0.4, x2[2]=0.3; intercept 0.9 + sample mean(u) = {0.9 + uj.mean():.3f}; "
          f"sample sd(u) = {uj.std(ddof=1):.3f}")

    # Cross-check: within-patient estimates from a fixed-effects (conditional) Poisson GLM.
    import statsmodels.api as sm
    Xfe = np.column_stack([Xs[:, 1:], pd.get_dummies(grp).to_numpy(float)])
    fe = sm.GLM(yy, Xfe, family=sm.families.Poisson(), offset=np.log(expo)).fit()
    print("fixed-effects Poisson:", dict(zip(nm[1:], fe.params[:3].round(4))))
