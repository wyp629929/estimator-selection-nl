"""Numerical checks for Section 3 of the manuscript.

Run: python3 theory/theory_checks.py
Each block prints PASS/FAIL for the claim it checks.
"""
import numpy as np
from scipy import stats, optimize
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
ok = lambda c: "PASS" if c else "FAIL"


# ---------------------------------------------------------------------------
# Lemma 2: h(a,t) = P(|eps + a| <= t) is even in a and nonincreasing in |a|
# (strictly, for strictly decreasing densities and t > 0).
# ---------------------------------------------------------------------------
def h(dist, a, t):
    return dist.cdf(t - a) - dist.cdf(-t - a)

for name, dist in [("normal", stats.norm()), ("laplace", stats.laplace()), ("t3", stats.t(3))]:
    a = np.linspace(0, 4, 401)
    good = True
    for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
        vals = h(dist, a, t)
        good &= np.all(np.diff(vals) < 0) and np.allclose(vals, h(dist, -a, t))
    print(f"[Lemma 2] {name:7s} strictly decreasing in |a| and even: {ok(good)}")


# ---------------------------------------------------------------------------
# Theorem 1: no crossing -> quantiles at every level agree with MSE
# (heteroscedastic Gaussian noise, which N+ allows).
# ---------------------------------------------------------------------------
N = 2_000_000
X = rng.standard_normal((N, 2))
eps = rng.standard_normal(N) * (0.5 + 0.5 * np.abs(X[:, 1]))
g1 = 0.3 * np.abs(X[:, 0])
g2 = g1 + 0.4 * (X[:, 0] > 0.5)                     # |g1| <= |g2|, strict on a set of positive mass
r1, r2 = eps + g1, eps + g2
levels = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
agree = all(np.quantile(np.abs(r1), a) < np.quantile(np.abs(r2), a) for a in levels)
print(f"[Thm 1]   strict quantile order at all levels {levels}: {ok(agree)}; "
      f"MSE {np.mean(r1**2):.4f} < {np.mean(r2**2):.4f}: {ok(np.mean(r1**2) < np.mean(r2**2))}")


# ---------------------------------------------------------------------------
# Remark 2: with bimodal noise the 'every alpha' conclusion fails.
# ---------------------------------------------------------------------------
e = rng.choice([-1.0, 1.0], size=N)
q04 = lambda r: np.quantile(np.abs(r), 0.4, method="inverted_cdf")
print(f"[Rem 2]   bimodal noise: Q0.4|r_M|={q04(e):.2f} > Q0.4|r_M'|={q04(e + 1):.2f} "
      f"although |g_M|<|g_M'|: {ok(q04(e) > q04(e + 1))}")


# ---------------------------------------------------------------------------
# Remark 3 (symmetry cannot be dropped): standardized Gamma(2) noise, constant shifts.
# ---------------------------------------------------------------------------
G2 = stats.gamma(2, loc=-np.sqrt(2), scale=1 / np.sqrt(2))
assert abs(G2.mean()) < 1e-12 and abs(G2.var() - 1) < 1e-12
Wshift = lambda D, b, a: 2 * optimize.brentq(lambda t: D.cdf(t - b) - D.cdf(-t - b) - (1 - a), 0, 50, xtol=1e-12)
q95 = G2.ppf(0.95)
bs = np.linspace((np.sqrt(2) - q95) / 2, 0, 9)
closed = all(abs(Wshift(G2, b, 0.05) - 2 * (q95 + b)) < 1e-8 for b in bs)
print(f"[Rem 3]   q={q95:.4f} > sqrt2: {ok(q95 > np.sqrt(2))}; W(b) = 2(q+b) on [(sqrt2-q)/2, 0]: {ok(closed)}; "
      f"b=-0.25: W={Wshift(G2, -0.25, 0.05):.3f} < W(0)={Wshift(G2, 0.0, 0.05):.3f}: {ok(Wshift(G2, -0.25, 0.05) < Wshift(G2, 0.0, 0.05))}")
print(f"[Rem 3]   alpha=0.2: W(0.27)={Wshift(G2, 0.27, 0.2):.3f} < W(0)={Wshift(G2, 0.0, 0.2):.3f}: "
      f"{ok(Wshift(G2, 0.27, 0.2) < Wshift(G2, 0.0, 0.2))}")
sym = all(Wshift(D, b, a) >= Wshift(D, 0.0, a) - 1e-9 and abs(Wshift(D, b, a) - Wshift(D, -b, a)) < 1e-8
          for D in (stats.norm(), stats.t(3, scale=1 / np.sqrt(3)), stats.laplace()) for a in (0.05, 0.1, 0.2)
          for b in (0.05, 0.27, 1.0))
print(f"[Rem 3]   symmetric unimodal noise: W(b) even and minimal at b=0 (normal, t3, Laplace): {ok(sym)}")


# ---------------------------------------------------------------------------
# Proposition 2(b): follow the constructive proof exactly, with population
# quantities computed from closed-form CDFs (no Monte Carlo).
# ---------------------------------------------------------------------------
Z = stats.norm()
H = lambda t: h(Z, 0.0, t)

def quantile(F, tau):
    return optimize.brentq(lambda t: F(t) - tau, 0.0, 50.0)

def construct(alpha, b, delta, p_min=1e-3):
    tau = 1 - alpha
    hb = lambda t: h(Z, b, t)
    W2 = 2 * quantile(hb, tau)
    best = None
    # proof: any p in (0, alpha) with h_b(H^{-1}(tau/(1-p))) < tau works;
    # among those with p >= p_min * alpha, keep the one with the largest width gap
    for p in np.linspace(p_min * alpha, alpha * 0.99, 300):
        if hb(quantile(H, tau / (1 - p))) >= tau:
            continue
        c = np.sqrt((b**2 + delta) / p) * 1.01     # p c^2 > b^2 + delta
        F1 = lambda t: (1 - p) * H(t) + p * h(Z, c, t)
        W1 = 2 * quantile(F1, tau)
        if best is None or W2 - W1 > best[3] - best[2]:
            best = (p, c, W1, W2, 1 + p * c**2, 1 + b**2)
    return best

for alpha in [0.05, 0.1, 0.2]:
    for b in [0.2, 1.0]:
        p, c, W1, W2, G1, G2 = construct(alpha, b, delta=1.0)
        good = (0 < p < alpha) and (c > b) and (W1 < W2) and (G1 - G2 > 1.0)
        print(f"[Prop 2b] alpha={alpha:.2f} b={b:.1f}: p={p:.4f} c={c:.2f} "
              f"W1={W1:.4f} < W2={W2:.4f}, G1-G2={G1 - G2:.3f} > 1: {ok(good)}")

# Proposition 2(b), explicit threshold pbar = 1 - (1-alpha)/H(q_b): W1 < W2 for every c when p < pbar;
# Remark (sharpness): for p > pbar the width ranks model 2 first once c is large; normal and Laplace noise.
for dname, D in [("normal", stats.norm()), ("laplace", stats.laplace())]:
    good_range = good_left = good_right = True
    for alpha in [0.05, 0.1, 0.2]:
        tau = 1 - alpha
        for b in [0.25, 0.5, 1.0, 2.0]:
            qb = quantile(lambda t: h(D, b, t), tau)
            pbar = 1 - tau / h(D, 0.0, qb)
            good_range &= 0 < pbar < alpha
            W1 = lambda p, c: 2 * quantile(lambda t: (1 - p) * h(D, 0.0, t) + p * h(D, c, t), tau)
            good_left &= all(W1(p, c) < 2 * qb for p in np.linspace(pbar * 0.02, pbar * 0.98, 6)
                             for c in [0.1, b, 3.0, 10.0, 40.0])
            good_right &= all(W1(p, 45.0) > 2 * qb for p in list(np.linspace(pbar, alpha, 7)[1:-1]) + [alpha, 1.5 * alpha])
    print(f"[Prop 2b] {dname:7s} 0 < pbar < alpha: {ok(good_range)}; W1 < W2 for all c when p < pbar: {ok(good_left)}; "
          f"W1 > W2 for large c when p > pbar: {ok(good_right)}")

# finite-sample: split-CP picks model 1, validation MSE picks model 2.
# p is kept >= alpha/4 so that n*p calibration points land in B; with a tiny p,
# validation MSE also misses the rare large errors in finite samples (it is only
# consistent as n -> infinity), whereas CP width misses them for every n.
alpha, b = 0.1, 1.0
p, c, W1, W2, *_ = construct(alpha, b, delta=1.0, p_min=0.25)
print(f"[Prop 2b] finite-sample config: p={p:.4f} c={c:.2f} W1={W1:.4f} W2={W2:.4f}")
n, reps = 5000, 300
pick_cp1 = pick_val2 = 0
k = int(np.ceil((n + 1) * (1 - alpha)))
for _ in range(reps):
    e = rng.standard_normal(n)
    inB = rng.uniform(size=n) < p
    r1, r2 = e + c * inB, e + b
    w1, w2 = np.sort(np.abs(r1))[k - 1], np.sort(np.abs(r2))[k - 1]
    pick_cp1 += w1 < w2
    pick_val2 += np.mean(r2**2) < np.mean(r1**2)
print(f"[Prop 2b] n={n}: P(CP picks globally worse model 1)={pick_cp1 / reps:.3f}, "
      f"P(val-MSE picks model 2)={pick_val2 / reps:.3f}: {ok(pick_cp1 / reps > 0.9 and pick_val2 / reps > 0.9)}")


# ---------------------------------------------------------------------------
# Proposition 2(d): C_A(xi_2) = P(|r_1| <= xi_2, X in A) decides the width.
# Covariates take K cells; each cell has its own noise law (normal or skewed
# standardized Gamma, with its own scale: no assumption N), its own g_1, g_2.
# A is a random set of cells; g_1 outside A is arbitrary. Closed-form CDFs.
# (i)  C_A(xi_2) >= tau  =>  W_1 <= W_2 for every choice of g_1 outside A.
# (ii) C_A(xi_2) <  tau  =>  W_{1_c} > W_2 once c is large (g_1 + c outside A).
# ---------------------------------------------------------------------------
noise_laws = [stats.norm(), stats.gamma(2, loc=-np.sqrt(2), scale=1 / np.sqrt(2))]   # normal; standardized Gamma(2), skewed

def cover(cells, t):                               # sum_k w_k P(|s_k eps_k + g_k| <= t)
    return sum(w * (D.cdf((t - g) / s) - D.cdf((-t - g) / s)) for w, D, s, g in cells)

def q_cells(cells, tau):
    return optimize.brentq(lambda t: cover(cells, t) - tau, 0.0, 1e7, xtol=1e-12)

n_suff = n_rev = 0
good_suff = good_rev = good_risk = True
for _ in range(6000):
    K = int(rng.integers(2, 7))
    w = rng.dirichlet(np.ones(K))
    Ds = [noise_laws[i] for i in rng.integers(0, 2, K)]
    s = rng.uniform(0.3, 2.0, K)
    g1, g2 = rng.normal(0, 1.5, K), rng.normal(0, 1.5, K)
    inA = rng.uniform(size=K) < 0.85
    if inA.all() or not inA.any():
        continue
    alpha = rng.choice([0.05, 0.1, 0.2]); tau = 1 - alpha
    xi2 = q_cells(list(zip(w, Ds, s, g2)), tau)
    CA = cover([(w[k], Ds[k], s[k], g1[k]) for k in range(K) if inA[k]], xi2)
    if abs(CA - tau) < 1e-6:
        continue
    if CA > tau:                                   # (i): any g_1 outside A, including huge ones
        n_suff += 1
        for _ in range(4):
            g1o = np.where(inA, g1, rng.normal(0, 1, K) * 10.0 ** rng.uniform(-1, 4))
            good_suff &= q_cells(list(zip(w, Ds, s, g1o)), tau) < xi2
    else:                                          # (ii): g_1 + c outside A
        n_rev += 1
        cells_c = lambda c: list(zip(w, Ds, s, np.where(inA, g1, g1 + c)))
        good_rev &= q_cells(cells_c(1e5), tau) > xi2
        risk = lambda c: sum(w * (s**2 + np.where(inA, g1, g1 + c) ** 2))
        good_risk &= risk(1e3) > risk(1e2) > risk(10.0)
print(f"[Prop 2d] (i)  C_A(xi2) > tau => W1 < W2 for any g1 outside A ({n_suff} configs, normal/skewed, "
      f"heteroscedastic): {ok(good_suff and n_suff > 100)}")
print(f"[Prop 2d] (ii) C_A(xi2) < tau => W_1c > W2 for large c ({n_rev} configs); G(1_c) grows: "
      f"{ok(good_rev and good_risk and n_rev > 200)}")

# Remark 4: in the construction of part (b), C_A(xi_2) = (1-p) H(q_b), and C_A > tau iff p < pbar.
good = True
for dname, D in [("normal", stats.norm()), ("laplace", stats.laplace()), ("t3", stats.t(3))]:
    for alpha in [0.05, 0.1, 0.2]:
        tau = 1 - alpha
        for b in [0.25, 0.5, 1.0, 2.0]:
            qb = quantile(lambda t: h(D, b, t), tau)
            pbar = 1 - tau / h(D, 0.0, qb)
            for p in np.linspace(0.001, 0.3, 60):
                cells = [(1 - p, D, 1.0, b), (p, D, 1.0, b)]            # model 2: g = b everywhere
                xi2 = q_cells(cells, tau)
                CA = (1 - p) * h(D, 0.0, xi2)
                good &= abs(xi2 - qb) < 1e-8 and ((CA > tau) == (p < pbar) or abs(p - pbar) < 1e-9)
print(f"[Rem 4]   part (b) is the case A = B^c: C_A(xi2) = (1-p)H(q_b) > tau iff p < pbar: {ok(good)}")


# ---------------------------------------------------------------------------
# Theorem 4 / Example 1: reflection T(x) = (-x1, x2, ..., xd).
# fhat_1 = 0, fhat_2 = c (both T-invariant), f0 = c*1{x1>0} vs its mirror.
# ---------------------------------------------------------------------------
d, c = 3, 1.5
f_hat = [lambda x: np.zeros(len(x)), lambda x: np.full(len(x), c)]
worlds = {"P": lambda x: c * (x[:, 0] > 0), "P^T": lambda x: c * (x[:, 0] < 0)}

def sample(f0, n):
    x = rng.standard_normal((n, d))
    y = f0(x) + rng.standard_normal(n)
    R = np.column_stack([y - fh(x) for fh in f_hat])
    return x, R

def local_oracle(f0, x):
    return np.argmin(np.column_stack([(f0(x) - fh(x))**2 for fh in f_hat]), axis=1)

# (i) the residual matrix has the same law in both worlds
xs = {w: sample(f0, 200_000) for w, f0 in worlds.items()}
ks = [stats.ks_2samp(xs["P"][1][:, m], xs["P^T"][1][:, m]).pvalue for m in range(2)]
diff_ks = stats.ks_2samp(xs["P"][1][:, 0] - xs["P"][1][:, 1], xs["P^T"][1][:, 0] - xs["P^T"][1][:, 1]).pvalue
print(f"[Thm 4i]  KS p-values, residual laws P vs P^T: model1={ks[0]:.3f} model2={ks[1]:.3f} "
      f"joint(r1-r2)={diff_ks:.3f}: {ok(min(ks + [diff_ks]) > 0.01)}")

# (iii) residual-only selectors: average misalignment over the two worlds >= 1/2
def sel_cp(x_test, R, alpha=0.1):                  # global, residual-only
    return np.full(len(x_test), np.argmin(np.quantile(np.abs(R), 1 - alpha, axis=0)))

def sel_val(x_test, R):                            # global, residual-only
    return np.full(len(x_test), np.argmin(np.mean(R**2, axis=0)))

def sel_xdep(x_test, R):                           # x-dependent, residual-only
    flip = np.mean(R[:, 0] - R[:, 1]) > 0.75       # any residual statistic
    return np.where((x_test[:, 0] > 0) ^ flip, 1, 0)

def sel_knn(x_test, R, x_cal):                     # conditional selector of Thm 4(iv)
    k = int(np.sqrt(len(x_cal)))
    eta = [KNeighborsRegressor(k).fit(x_cal, R[:, m]**2).predict(x_test) for m in range(2)]
    return np.argmin(np.column_stack(eta), axis=1)

n_cal, n_test, reps = 2000, 2000, 40
res = {s: {w: [] for w in worlds} for s in ["CP", "val", "x-dep", "kNN"]}
for _ in range(reps):
    for w, f0 in worlds.items():
        x_cal, R = sample(f0, n_cal)
        x_test = rng.standard_normal((n_test, d))
        orc = local_oracle(f0, x_test)
        res["CP"][w].append(np.mean(sel_cp(x_test, R) != orc))
        res["val"][w].append(np.mean(sel_val(x_test, R) != orc))
        res["x-dep"][w].append(np.mean(sel_xdep(x_test, R) != orc))
        res["kNN"][w].append(np.mean(sel_knn(x_test, R, x_cal) != orc))
for s, v in res.items():
    dP, dT = np.mean(v["P"]), np.mean(v["P^T"])
    if s == "kNN":
        print(f"[Thm 4iv] {s:6s} Delta_loc: P={dP:.3f} P^T={dT:.3f}  (conditional selector -> small): {ok(max(dP, dT) < 0.1)}")
    else:
        print(f"[Thm 4iii] {s:5s} Delta_loc: P={dP:.3f} P^T={dT:.3f}  avg={(dP + dT) / 2:.3f} >= 0.5: "
              f"{ok((dP + dT) / 2 >= 0.5 - 0.02)}")


# ---------------------------------------------------------------------------
# Lemma 3 and Proposition 3 (binary outcomes). Population claims are checked
# exactly: X takes finitely many values, every probability and every value is
# a Fraction, so quantiles and boundary cases (F(xi) = tau exactly, |r| = 1/2)
# involve no rounding.
# ---------------------------------------------------------------------------
from fractions import Fraction as Fr

def frac(lo, hi, den=48):
    return Fr(int(rng.integers(int(lo * den), int(hi * den) + 1)), den)

def q_atoms(atoms, tau):
    """Q_tau(Z) = inf{t >= 0: P(Z <= t) >= tau} for Z given as [(value, mass), ...]."""
    cum = Fr(0)
    for v, m in sorted(atoms):
        cum += m
        if cum >= tau:
            return v
    raise ValueError

def alter_above(atoms, cut, lo, strict):
    """Split every atom with value above cut (strict) or >= cut into two atoms at new values >= lo (> lo if strict)."""
    out = []
    for v, m in atoms:
        if (v > cut) if strict else (v >= cut):
            for _ in range(2):
                w = lo + frac(0, 3)
                out.append((w + Fr(1, 96) if strict else w, m / 2))
        else:
            out.append((v, m))
    return out

def random_masses(K):
    w = [Fr(int(rng.integers(1, 7))) for _ in range(K)]
    return [x / sum(w) for x in w]

# Lemma 3(i): population. tau is often set to an attained cumulative mass.
good, boundary = True, 0
for _ in range(3000):
    K = int(rng.integers(2, 7))
    atoms = list(zip([frac(0, 3) for _ in range(K)], random_masses(K)))
    if rng.random() < 0.5:                                    # tau equal to a cumulative mass
        ms = [m for _, m in sorted(atoms)]
        tau = sum(ms[:int(rng.integers(1, K + 1))])
        tau = tau if tau < 1 else Fr(1, 2)
        boundary += 1
    else:
        tau = frac(0.02, 0.98, 97)
    xi = q_atoms(atoms, tau)
    good &= q_atoms(alter_above(atoms, xi, xi, strict=True), tau) == xi
print(f"[Lem 3i]  Q_tau unchanged after moving mass above xi to other values above xi "
      f"(3000 cases, {boundary} with F(xi)=tau exactly): {ok(good)}")

# Lemma 3 for real-valued scores (e.g. CQR scores, which can be negative): the same two statements.
good_r = True
for _ in range(3000):
    n = int(rng.integers(5, 60))
    s = rng.integers(-6, 6, n).astype(float) + rng.choice([0.0, 0.5], n)      # negative values and ties
    tau = rng.uniform(0.05, 0.95)
    k = int(np.ceil((n + 1) * tau))
    if k > n:
        continue
    xih = np.sort(s)[k - 1]
    s2 = np.where(s > xih, xih + rng.uniform(1e-9, 50, n), s)
    good_r &= np.sort(s2)[k - 1] == xih
atoms_ok = True
for _ in range(2000):
    K = int(rng.integers(2, 7))
    atoms = list(zip([frac(-3, 3) for _ in range(K)], random_masses(K)))
    tau = frac(0.02, 0.98, 97)
    cum, xi = Fr(0), None
    for v, m in sorted(atoms):
        cum += m
        if cum >= tau:
            xi = v; break
    moved = [(xi + Fr(1, 96) + frac(0, 3), m) if v > xi else (v, m) for v, m in atoms]
    cum2, xi2 = Fr(0), None
    for v, m in sorted(moved):
        cum2 += m
        if cum2 >= tau:
            xi2 = v; break
    atoms_ok &= xi2 == xi
print(f"[Lem 3]   real-valued scores: k-th order statistic unchanged: {ok(good_r)}; population quantile unchanged: {ok(atoms_ok)}")

# Lemma 3(ii): finite sample, with ties.
good_q, good_n = True, True
for _ in range(3000):
    n = int(rng.integers(5, 60))
    s = rng.integers(0, 8, n).astype(float)                   # many ties
    tau = rng.uniform(0.05, 0.95)
    k = int(np.ceil((n + 1) * tau))
    if k > n:
        continue
    xih = np.sort(s)[k - 1]
    s2 = np.where(s > xih, xih + rng.uniform(1e-9, 50, n), s)
    good_q &= np.sort(s2)[k - 1] == xih
    good_n &= np.sum(s > xih) <= n - k
print(f"[Lem 3ii] k-th order statistic unchanged: {ok(good_q)}; at most n-k values above it: {ok(good_n)}")


def binary_atoms(w, p, fhat):
    """Atoms of |r| = |Y - fhat(X)| and the joint event masses, for X on K points."""
    pos = [(abs(1 - f), wk * pk) for wk, pk, f in zip(w, p, fhat)]
    neg = [(abs(f), wk * (1 - pk)) for wk, pk, f in zip(w, p, fhat)]
    return pos, neg

def random_binary(K, f_lo=-0.5, f_hi=1.5):
    w = random_masses(K)
    p = [frac(0, 1) for _ in range(K)]
    fhat = [frac(f_lo, f_hi) for _ in range(K)]
    return w, p, fhat

# Proposition 3(a): P(Y=1, |r| <= xi) >= pi - alpha whenever pi > alpha.
good, cases = True, 0
for _ in range(4000):
    w, p, fhat = random_binary(int(rng.integers(1, 7)))
    pi = sum(wk * pk for wk, pk in zip(w, p))
    alpha = frac(0.01, 0.6)
    if not (0 < pi < 1 and pi > alpha):
        continue
    pos, neg = binary_atoms(w, p, fhat)
    xi = q_atoms(pos + neg, 1 - alpha)
    good &= sum(m for v, m in pos if v <= xi) >= pi - alpha
    cases += 1
print(f"[Prop 3a] P(Y=1,|r|<=xi) >= pi-alpha ({cases} cases): {ok(good)}")

# Proposition 3(b).
good_xi = good_alt = good_formula = good_pi = True
cases = sub = 0
for _ in range(6000):
    K = int(rng.integers(1, 7))
    w, p, fhat = random_binary(K, -0.75, 1.25)
    if rng.random() < 0.5:                                    # make positives predicted below 1/2
        fhat = [min(f, Fr(1, 2) - Fr(1, 48)) if pk > 0 else f for f, pk in zip(fhat, p)]
    pi = sum(wk * pk for wk, pk in zip(w, p))
    pos, neg = binary_atoms(w, p, fhat)
    e = sum(m for v, m in pos + neg if v >= Fr(1, 2))
    alpha = frac(0.01, 0.6)
    if not (0 < pi < 1 and e < alpha):
        continue
    cases += 1
    xi = q_atoms(pos + neg, 1 - alpha)
    good_xi &= xi < Fr(1, 2)
    good_alt &= q_atoms(alter_above(pos + neg, Fr(1, 2), Fr(1, 2), strict=False), 1 - alpha) == xi
    if all(f < Fr(1, 2) for f, pk in zip(fhat, p) if pk > 0):
        sub += 1
        good_pi &= pi < alpha
        tau0 = (1 - alpha) / (1 - pi)
        neg0 = [(v, m / (1 - pi)) for v, m in neg if m > 0]
        good_formula &= q_atoms(neg0, tau0) == xi
print(f"[Prop 3b] e<alpha => xi<1/2 ({cases} cases): {ok(good_xi)}; width unchanged when residuals >=1/2 "
      f"are moved anywhere >=1/2: {ok(good_alt)}")
print(f"[Prop 3b] positives predicted <1/2 ({sub} cases): pi<alpha: {ok(good_pi)}; "
      f"W = 2 Q_tau'(|fhat| | Y=0): {ok(good_formula)}")
good = True
for _ in range(2000):
    w, p, _ = random_binary(int(rng.integers(1, 7)))
    pi = sum(wk * pk for wk, pk in zip(w, p))
    alpha = frac(0.01, 0.6)
    if not (0 < pi < alpha):
        continue
    pos, neg = binary_atoms(w, p, [Fr(0)] * len(w))
    good &= q_atoms(pos + neg, 1 - alpha) == 0
print(f"[Prop 3b] fhat = 0 and pi<alpha gives W = 0: {ok(good)}")

# Proposition 3(c).
good_xi = good_w = good_brier = True
cases = with_p = 0
for _ in range(3000):
    K = int(rng.integers(1, 7))
    w, p, fhat = random_binary(K)
    use_p = rng.random() < 0.5
    if use_p:
        fhat = list(p)
    pi = sum(wk * pk for wk, pk in zip(w, p))
    pos, neg = binary_atoms(w, p, fhat)
    e = sum(m for v, m in pos + neg if v >= Fr(1, 2))
    alpha = frac(0.01, 0.6)
    if not (0 < pi < 1 and e > alpha):
        continue
    cases += 1
    xi = q_atoms(pos + neg, 1 - alpha)
    good_xi &= xi >= Fr(1, 2)
    cs = [1 - xi, xi, Fr(1, 2)] + [1 - xi + (2 * xi - 1) * frac(0, 1) for _ in range(3)]
    for c in cs:
        for lam in [Fr(0), Fr(1), frac(0, 1), frac(0, 1)]:
            ft = [c + lam * (f - c) for f in fhat]
            pos2, neg2 = binary_atoms(w, p, ft)
            good_w &= q_atoms(pos2 + neg2, 1 - alpha) <= xi
            if use_p:
                brier = lambda g: sum(wk * (pk * (1 - gk) ** 2 + (1 - pk) * gk ** 2) for wk, pk, gk in zip(w, p, g))
                gap = brier(ft) - brier(p)
                good_brier &= gap == (1 - lam) ** 2 * sum(wk * (pk - c) ** 2 for wk, pk in zip(w, p))
    with_p += use_p
print(f"[Prop 3c] e>alpha => xi>=1/2 ({cases} cases): {ok(good_xi)}; shrinking towards c in [1-xi,xi] "
      f"never widens: {ok(good_w)}")
print(f"[Prop 3c] fhat=p ({with_p} cases): Brier gap = (1-lambda)^2 E(p-c)^2 exactly: {ok(good_brier)}")

# Remark (two regimes): predictions in (-1/2, 1/2) give e = pi exactly.
good = True
for _ in range(2000):
    w, p, fhat = random_binary(int(rng.integers(1, 7)), -0.47, 0.47)
    pos, neg = binary_atoms(w, p, fhat)
    good &= sum(m for v, m in pos + neg if v >= Fr(1, 2)) == sum(wk * pk for wk, pk in zip(w, p))
print(f"[Rem]     predictions in (-1/2,1/2) give e = pi: {ok(good)}")


# ---------------------------------------------------------------------------
# Theorem 4(iv) under the relaxed moment condition E[(r_M^2 - r_1^2)^2] < inf:
# t3 noise (finite variance, no fourth moment), same reflection example.
# (a) kNN on r_M^2 and kNN on r_M^2 - r_1^2 pick the same model at every point;
# (b) the misalignment of the kNN selector still decreases towards 0.
# ---------------------------------------------------------------------------
def sample_t3(f0, n):
    x = rng.standard_normal((n, d))
    y = f0(x) + rng.standard_t(3, n) / np.sqrt(3)
    return x, np.column_stack([y - 0.0, y - c])

same, dl = True, {}
for n_cal in (500, 2000, 8000):
    errs = []
    for _ in range(20):
        x_cal, R = sample_t3(worlds["P"], n_cal)
        x_test = rng.standard_normal((2000, d))
        k = int(np.sqrt(n_cal))
        nn = KNeighborsRegressor(k).fit(x_cal, R**2)
        eta = nn.predict(x_test)
        diff = KNeighborsRegressor(k).fit(x_cal, R[:, 1]**2 - R[:, 0]**2).predict(x_test)
        s_eta, s_diff = np.argmin(eta, axis=1), np.where(diff < 0, 1, 0)
        same &= bool(np.all(s_eta == s_diff) or np.mean(s_eta != s_diff) < 1e-3)
        errs.append(np.mean(s_eta != local_oracle(worlds["P"], x_test)))
    dl[n_cal] = np.mean(errs)
print(f"[Thm 4iv] t3 noise: kNN on r^2 and on r_M^2 - r_1^2 pick the same model: {ok(same)}; "
      f"Delta_loc by n {dict((k, round(v, 3)) for k, v in dl.items())} decreasing: {ok(dl[500] > dl[2000] > dl[8000])}")


# ---------------------------------------------------------------------------
# Corollary 3: over measure-preserving maps, no residual-only rule beats a single model.
# X ~ Unif[0,1), N(0,1) noise independent of X, T_u(x) = (x + u) mod 1 (phi = identity).
# Bounds: min_M G(M) - E[min_M D(M|X)] (local regret), 1 - max_M P{M in A(X)} (misalignment).
# ---------------------------------------------------------------------------
grid = (np.arange(4000) + 0.5) / 4000
f0c = lambda x: 1.5 * np.sin(2 * np.pi * x)
fhc = [lambda x: np.zeros_like(x), lambda x: 1.0 * (x > 0.3) + 0.5 * (x > 0.75) - 0.2]
gsq = np.column_stack([(f0c(grid) - fh(grid)) ** 2 for fh in fhc])          # D(M|x) - sigma^2
rho_g = gsq - gsq.min(axis=1, keepdims=True)
bound_reg = rho_g.mean(axis=0).min()
oracle_g = gsq.argmin(axis=1)
bound_mis = 1 - max(np.mean(oracle_g == m) for m in range(2))
idx = lambda x: np.minimum((x * len(grid)).astype(int), len(grid) - 1)

def reg_mis(rule, u, n=2000, reps=40):          # local regret and misalignment of a residual-only rule under P^(T_u)
    regs, mis = [], []
    for _ in range(reps):
        xc = rng.uniform(size=n); xt = (xc + u) % 1
        y = f0c(xt) + rng.standard_normal(n)
        R = np.column_stack([y - fh(xt) for fh in fhc])
        s = rule(grid, R)                       # selection at every test point of the grid
        k = idx((grid + u) % 1)                 # D_{P^(T)}(M|x) = D_P(M|T x)
        regs.append(rho_g[k, s].mean()); mis.append(np.mean(s != oracle_g[k]))
    return np.mean(regs), np.mean(mis)

rules = {
    "oracle of P (x-dependent)": lambda x, R: oracle_g[idx(x)],
    "residual-switched oracle":   lambda x, R: np.where(np.mean(R[:, 0] ** 2 - R[:, 1] ** 2) > 0, oracle_g[idx(x)], 1 - oracle_g[idx(x)]),
    "CP width (global)":          lambda x, R: np.full(len(x), np.argmin(np.quantile(np.abs(R), 0.9, axis=0))),
}
us = np.arange(32) / 32
good_max = good_avg = True
for name, rule in rules.items():
    rm = np.array([reg_mis(rule, u, reps=8) for u in us])
    good_max &= rm[:, 0].max() >= bound_reg - 0.01 and rm[:, 1].max() >= bound_mis - 0.01
    good_avg &= rm[:, 0].mean() >= bound_reg - 0.01 and rm[:, 1].mean() >= bound_mis - 0.01
print(f"[Cor 3]   bounds: local regret {bound_reg:.4f}, misalignment {bound_mis:.4f}; "
      f"max over 32 shifts >= bound for 3 residual-only rules: {ok(good_max)}; average over shifts >= bound: {ok(good_avg)}")

# E[m_L] -> bound as L grows (random shifts), regret version
mL = {}
for L in (1, 4, 32, 256):
    V = rng.uniform(size=(400, L)); X0 = rng.uniform(size=400)
    k = idx((X0[:, None] + V) % 1)
    mL[L] = np.mean(rho_g[k].mean(axis=1).min(axis=1))
print(f"[Cor 3]   E[m_L] by L {dict((k, round(float(v), 4)) for k, v in mL.items())} -> {bound_reg:.4f}: "
      f"{ok(mL[1] < mL[256] and abs(mL[256] - bound_reg) < 0.03 * max(bound_reg, 1e-9) + 0.01)}")

# validation MSE attains the regret bound under every shift
val = lambda x, R: np.full(len(x), np.argmin(np.mean(R ** 2, axis=0)))
vr = [reg_mis(val, u, n=20000, reps=10)[0] for u in (0.0, 0.17, 0.5, 0.83)]
print(f"[Cor 3]   validation MSE local regret under 4 shifts {np.round(vr, 4)} = bound {bound_reg:.4f}: "
      f"{ok(max(abs(v - bound_reg) for v in vr) < 0.01)}")


# ---------------------------------------------------------------------------
# Decomposition (eq:decomposition): Reg_loc(S) = Reg(S) + V for global S, V = min G - E min D.
# Same Unif[0,1) configuration as Corollary 3; CP width at n=250 (often wrong) and val MSE.
# Plus: the kNN conditional selector's local regret tends to zero.
# ---------------------------------------------------------------------------
Gc = gsq.mean(axis=0) + 1.0                       # global risks (sigma^2 = 1)
Vc = Gc.min() - (gsq.min(axis=1) + 1.0).mean()
# A global rule is a mixture of constant rules, and both sides of the identity are linear in that
# mixture, so checking every constant rule and the random choices of CP width and val MSE suffices.
good_dec, picks = True, []
for s in range(2):
    good_dec &= abs(rho_g[:, s].mean() - ((Gc[s] - Gc.min()) + Vc)) < 1e-12
for _ in range(300):
    xc = rng.uniform(size=40)
    y = f0c(xc) + rng.standard_normal(40)
    R = np.column_stack([y - fh(xc) for fh in fhc])
    for s in (np.argmin(np.quantile(np.abs(R), 0.9, axis=0)), np.argmin(np.mean(R ** 2, axis=0))):
        good_dec &= abs(rho_g[:, s].mean() - ((Gc[s] - Gc.min()) + Vc)) < 1e-12
        picks.append(int(s))
print(f"[Eq 4]    V={Vc:.4f} (= Cor 3 bound {bound_reg:.4f}: {ok(abs(Vc - bound_reg) < 1e-12)}); "
      f"Reg_loc = Reg + V for both constant rules and for CP width and val MSE at n=40 "
      f"(model 1 picked in {np.mean(np.array(picks) == 1):.1%} of draws): {ok(good_dec and 0 < np.mean(np.array(picks) == 1) < 1)}")

knn_reg = {}
for n_cal in (500, 4000, 32000):
    vals = []
    for _ in range(5):
        xc = rng.uniform(size=n_cal)
        y = f0c(xc) + rng.standard_normal(n_cal)
        R = np.column_stack([y - fh(xc) for fh in fhc])
        eta = KNeighborsRegressor(int(np.sqrt(n_cal))).fit(xc[:, None], R ** 2).predict(grid[:, None])
        vals.append(rho_g[np.arange(len(grid)), eta.argmin(axis=1)].mean())
    knn_reg[n_cal] = float(np.mean(vals))
print(f"[Eq 4]    kNN local regret by n {dict((k, round(v, 4)) for k, v in knn_reg.items())} vs V={Vc:.4f}: "
      f"{ok(knn_reg[500] > knn_reg[4000] > knn_reg[32000] and knn_reg[32000] < 0.2 * Vc)}")

# Torus [0,1)^2 with translations only: smooth periodic f0, fhat; E[m_L] -> V.
f02 = lambda x: np.sin(2 * np.pi * x[:, 0]) + 0.8 * np.cos(2 * np.pi * x[:, 1])
fh2 = [lambda x: np.zeros(len(x)), lambda x: 0.6 * np.sin(2 * np.pi * (x[:, 0] + x[:, 1]))]
g2sq = lambda x: np.column_stack([(f02(x) - fh(x)) ** 2 for fh in fh2])
Xg = rng.uniform(size=(200_000, 2)); gg = g2sq(Xg)
V2 = gg.mean(axis=0).min() - gg.min(axis=1).mean()
X0, Lm = rng.uniform(size=(300, 2)), {}
for L in (1, 8, 128):
    U = rng.uniform(size=(L, 2))
    vals = []
    for x in X0:
        gx = g2sq((x + U) % 1)
        rx = gx - gx.min(axis=1, keepdims=True)
        vals.append(rx.mean(axis=0).min())
    Lm[L] = float(np.mean(vals))
print(f"[Cor 3]   torus [0,1)^2, translations only: E[m_L] by L {dict((k, round(v, 4)) for k, v in Lm.items())} -> V={V2:.4f}: "
      f"{ok(Lm[1] < Lm[8] < Lm[128] + 1e-9 and abs(Lm[128] - V2) < 0.05 * V2)}")
