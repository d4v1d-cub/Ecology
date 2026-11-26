"""Empirical mean and mode of the pairwise-distance histograms (sigma = 0).
File format: '# ...' header, then columns bin_left bin_right bin_center count density.
Mean: over all pairs (including the d~0 bin = runs that reached the same fixed point).
Mode: maximum of the histogram smoothed with a Gaussian kernel (bandwidth 0.01), EXCLUDING the
first bin; the range over bandwidths {0.005, 0.01, 0.02} is stored as mode_lo/mode_hi.
Writes results/histograms_empirical.json."""
import glob, re, os, json, numpy as np
from config import HIST_DIR, RESULTS_DIR
MUS = {0.37, 0.38}          # the sigma=0 values of mu that overlap with the vectors in the notes
NS = {1024, 8192}
BWS, BW0 = (0.005, 0.01, 0.02), 0.01
out = []
for f in sorted(glob.glob(os.path.join(HIST_DIR, '*sigma_0.000*pairwise_dist_histogram*.txt'))):
    mu = float(re.search(r'mu_([0-9.]+)_', f).group(1)); N = int(re.search(r'_N_(\d+)_', f).group(1))
    if round(mu, 3) not in MUS or N not in NS: continue
    A = np.loadtxt(f, comments='#'); c, n = A[:, 2], A[:, 3]; tot = n.sum()
    mean = (c * n).sum() / tot; sd = np.sqrt(((c - mean)**2 * n).sum() / tot)
    modes = {}
    for bw in BWS:
        k = np.exp(-0.5 * ((c[:, None] - c[None, 1:]) / bw)**2)
        modes[bw] = float(c[np.argmax((k * n[None, 1:]).sum(1))])
    out.append(dict(mu=mu, N=N, mean=float(mean), sd=float(sd), mode=modes[BW0], mode_lo=min(modes.values()),
                    mode_hi=max(modes.values()), p_same=float(n[0] / tot),
                    mean_excl0=float((c[1:] * n[1:]).sum() / n[1:].sum()),
                    dmax_obs=float(c[np.nonzero(n)[0][-1]]), pairs=float(tot), file=os.path.basename(f)))
    print({k: v for k, v in out[-1].items() if k != 'file'})
if not out: raise SystemExit(f"No matching histograms in {HIST_DIR} (set HIST_DIR)")
json.dump(out, open(os.path.join(RESULTS_DIR, 'histograms_empirical.json'), 'w'), indent=1)
