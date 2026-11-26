"""Per-vector sample mean/variance -> Beta fit -> null-model predictions (Table 2 / Figure 3 of the notes).
Also: distribution-free check (empirical central moments), Gaussian-Q comparison,
point-mass fractions and the resulting cap d_max = sqrt(1 - p0^2 - p1^2), observed pairwise distances.
Writes results/vectors_per_file.json and results/vectors_by_mu.json."""
import glob, re, os, json, itertools, numpy as np
from null_model import beta_params, predictions_beta, predictions_empirical, predictions, gaussian_central_moments
from config import VECTORS_DIR, RESULTS_DIR

EPS = 1e-6          # threshold for "exactly 0" / "exactly 1"
S2_ZERO = 1e-10     # below this the vector is treated as homogeneous (mu=0.350: s2 ~ 1e-12)
files = sorted(glob.glob(os.path.join(VECTORS_DIR, '*.txt')))
if not files: raise SystemExit(f"No vector files in {VECTORS_DIR} (set VECTORS_DIR)")
rows, vecs = [], {}
for f in files:
    mu = float(re.search(r'mu_([0-9.]+)_', f).group(1)); N = int(re.search(r'_N_(\d+)_', f).group(1))
    seed = int(re.search(r'seedinit_(\d+)', f).group(1))
    x = np.loadtxt(f, comments='#')[:, 1]          # column av(n); column var(n) is ignored
    vecs.setdefault((mu, N), []).append(x)
    mbar, s2 = x.mean(), x.var(ddof=1)
    r = dict(mu=mu, N=N, seed=seed, mbar=mbar, s2=s2, p0=(x < EPS).mean(), p1=(x > 1 - EPS).mean())
    if s2 > S2_ZERO:
        al, be, nu = beta_params(mbar, s2); r.update(alpha=al, beta=be)
        r.update(predictions_beta(N, mbar, s2))
        e = predictions_empirical(x); r.update(emp_mean_d=e['mean_d'], emp_mode_d=e['mode_d'])
        g = predictions(N, s2, *gaussian_central_moments(s2)); r.update(gauss_mean_d=g['mean_d'], gauss_mode_d=g['mode_d'])
    else:
        r.update(alpha=np.inf, beta=np.inf, mean_d2=0, var_d2=0, mode_d2=0, mean_d=0, var_d=0, mode_d=0,
                 emp_mean_d=0, emp_mode_d=0, gauss_mean_d=0, gauss_mode_d=0)
    rows.append({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()})

agg = []
for (mu, N), vs in sorted(vecs.items()):
    g = [r for r in rows if r['mu'] == mu and r['N'] == N]
    F = lambda k: np.array([r[k] for r in g], float)
    fin = lambda k: float(np.mean(F(k)[np.isfinite(F(k))])) if np.isfinite(F(k)).any() else float('inf')
    diff = F('mode_d') - F('mean_d')
    x = np.concatenate(vs); p0, p1 = (x < EPS).mean(), (x > 1 - EPS).mean()
    agg.append(dict(mu=mu, N=N, mbar=F('mbar').mean(), s2=F('s2').mean(), s2sd=F('s2').std(ddof=1),
                    alpha=fin('alpha'), beta=fin('beta'),
                    md=F('mean_d').mean(), mdsd=F('mean_d').std(ddof=1), mo=F('mode_d').mean(), mosd=F('mode_d').std(ddof=1),
                    diff=diff.mean(), diffsd=diff.std(ddof=1), sd_d=float(np.mean(np.sqrt(F('var_d')))),
                    p0=p0, p1=p1, dmax_cap=float(np.sqrt(1 - p0**2 - p1**2)),
                    pairs=[float(np.sqrt(((a - b)**2).mean())) for a, b in itertools.combinations(vs, 2)]))
json.dump(rows, open(os.path.join(RESULTS_DIR, 'vectors_per_file.json'), 'w'), indent=1)
json.dump(agg, open(os.path.join(RESULTS_DIR, 'vectors_by_mu.json'), 'w'), indent=1, default=float)

print(f"{'mu':>6}{'N':>6}{'mbar':>9}{'s2':>11}{'alpha':>9}{'beta':>9}{'<d>':>9}{'d_mode':>9}{'Dx1e4':>8}{'sd(d)':>8}{'p0':>7}{'p1':>7}{'dmax':>7}")
for a in agg:
    print(f"{a['mu']:6.3f}{a['N']:6d}{a['mbar']:9.5f}{a['s2']:11.3e}{a['alpha']:9.3g}{a['beta']:9.3g}{a['md']:9.4f}"
          f"{a['mo']:9.4f}{1e4*a['diff']:8.2f}{a['sd_d']:8.4f}{a['p0']:7.3f}{a['p1']:7.3f}{a['dmax_cap']:7.3f}")
print("\nBeta vs empirical-moments vs Gaussian (per file): <d>, d_mode")
for r in rows:
    print(f"{r['mu']:.3f} {r['N']:5d} s{r['seed']}: <d> {r['mean_d']:.5f} {r['emp_mean_d']:.5f} {r['gauss_mean_d']:.5f} | "
          f"mode {r['mode_d']:.5f} {r['emp_mode_d']:.5f} {r['gauss_mode_d']:.5f}")
