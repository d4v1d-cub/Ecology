"""
Null model for the distance between two independent vectors with i.i.d. components.

Model:  d^2 = (1/N) sum_i (m_i - m_hat_i)^2, with m_i, m_hat_i i.i.d. ~ Q(m).
Q is a Beta(alpha, beta) distribution matched to the sample mean and variance,
or (distribution-free variant) characterised directly by its central moments.

All formulas are those of the notes (counting_fixed_points.tex):
  kappa_1(X) = 2 s^2                                   (X = D^2, D = m - m_hat)
  kappa_2(X) = 2 mu4 + 2 s^4
  kappa_3(X) = 2 mu6 + 18 mu4 s^2 - 20 mu3^2 - 20 s^6
  <d^2>       = 2 s^2                      (exact, any Q, any N)
  Var(d^2)    = kappa_2 / N
  (d^2)_mode  = 2 s^2 - kappa_3 / (2 N kappa_2)            (Edgeworth, o(1/N))
  <d>         = sqrt(2 s^2) [1 - kappa_2 / (8 N (2 s^2)^2)] (delta method, O(1/N^2))
  Var(d)      = kappa_2 / (4 N 2 s^2)
  d_mode^2    = 2 s^2 + [kappa_2/(2*2 s^2) - kappa_3/(2 kappa_2)] / N
"""
import numpy as np
from math import comb


# ---------------------------------------------------------------- Beta helpers
def beta_params(mbar, s2):
    """Method of moments: (alpha, beta, nu=alpha+beta). Requires 0 < s2 < mbar(1-mbar)."""
    w = mbar * (1 - mbar)
    nu = w / s2 - 1
    return mbar * nu, (1 - mbar) * nu, nu


def beta_central_moments(mbar, s2):
    """Central moments mu3, mu4, mu6 of the Beta matched to (mbar, s2), written with
    w = mbar(1-mbar) and nu = alpha+beta (numerically stable for huge nu)."""
    w = mbar * (1 - mbar)
    nu = w / s2 - 1
    mu3 = 2 * w * (1 - 2 * mbar) / ((nu + 1) * (nu + 2))
    mu4 = 3 * w * (w * (nu - 6) + 2) / ((nu + 1) * (nu + 2) * (nu + 3))
    mu6 = 5 * w * (w**2 * (3 * nu**2 - 86 * nu + 120) + w * (26 * nu - 120) + 24) / (
        (nu + 1) * (nu + 2) * (nu + 3) * (nu + 4) * (nu + 5))
    return mu3, mu4, mu6


def gaussian_central_moments(s2):
    """Central moments of a Gaussian with variance s2 (for the 'Beta vs Gaussian' check)."""
    return 0.0, 3 * s2**2, 15 * s2**3


def beta_ED2n(alpha, beta, n):
    """E[D^{2n}] for D = m - m_hat, m, m_hat i.i.d. Beta(alpha, beta) (exact)."""
    raw = lambda j: np.prod([(alpha + r) / (alpha + beta + r) for r in range(j)]) if j > 0 else 1.0
    return sum((-1)**j * comb(2 * n, j) * raw(j) * raw(2 * n - j) for j in range(2 * n + 1))


# ---------------------------------------------------------------- predictions
def cumulants_X(s2, mu3, mu4, mu6):
    """Cumulants of X = D^2 from the central moments of Q (valid for ANY Q)."""
    k1 = 2 * s2
    k2 = 2 * mu4 + 2 * s2**2
    k3 = 2 * mu6 + 18 * mu4 * s2 - 20 * mu3**2 - 20 * s2**3
    return k1, k2, k3


def predictions(N, s2, mu3, mu4, mu6):
    """Mean / variance / mode of d^2 and of d for vectors of length N."""
    if s2 <= 0:
        return dict(mean_d2=0.0, var_d2=0.0, mode_d2=0.0, mean_d=0.0, var_d=0.0, mode_d=0.0, k2=0.0, k3=0.0)
    k1, k2, k3 = cumulants_X(s2, mu3, mu4, mu6)
    return dict(
        mean_d2=k1,
        var_d2=k2 / N,
        mode_d2=k1 - k3 / (2 * N * k2),
        mean_d=np.sqrt(k1) * (1 - k2 / (8 * N * k1**2)),
        var_d=k2 / (4 * N * k1),
        mode_d=np.sqrt(k1 + k2 / (2 * N * k1) - k3 / (2 * N * k2)),
        k2=k2, k3=k3)


def predictions_beta(N, mbar, s2):
    return predictions(N, s2, *beta_central_moments(mbar, s2))


def predictions_empirical(x):
    """Distribution-free version: uses the sample central moments of the vector x."""
    c = x - x.mean()
    return predictions(len(x), x.var(ddof=1), (c**3).mean(), (c**4).mean(), (c**6).mean())


# ---------------------------------------------------------------- pair function I(D)
def I_closed_form(D, alpha, beta):
    """I(D) = int_D^1 Q(m) Q(m-D) dm via the Appell F1 closed form
    I = D^(2a-1)(1-D)^(a+b-1)/B(a,b) * F1(a; 2a+2b-2, 1-b; a+b; 1-D, 1-D^2).
    Falls back to the Euler integral of F1 when mpmath's series does not converge."""
    from mpmath import appellf1, beta as B, quad
    a, b, t = alpha, beta, D
    pref = (1 - t)**(a + b - 1) * t**(2 * a - 1) / B(a, b)
    try:
        return float(pref * appellf1(a, 2 * a + 2 * b - 2, 1 - b, a + b, 1 - t, 1 - t**2))
    except Exception:
        F = quad(lambda u: u**(a - 1) * (1 - u)**(b - 1) * (1 - (1 - t) * u)**(-(2 * a + 2 * b - 2))
                 * (1 - (1 - t**2) * u)**(b - 1), [0, 1]) / B(a, b)
        return float(pref * F)


# ---------------------------------------------------------------- exact P(d^2)
def phi_factory(alpha, beta, n=120):
    """Characteristic function phi(k) = E exp(i k D^2), by Gauss-Jacobi quadrature over (m, m_hat)."""
    from scipy.special import roots_jacobi
    t, w = roots_jacobi(n, beta - 1, alpha - 1)
    x = (t + 1) / 2
    w = w / w.sum()
    d2 = ((x[:, None] - x[None, :])**2).ravel()
    ww = (w[:, None] * w[None, :]).ravel()

    def phi(k):
        k = np.asarray(k, float)
        out = np.empty(len(k), complex)
        for i in range(0, len(k), 200):
            out[i:i + 200] = np.exp(1j * k[i:i + 200, None] * d2[None]) @ ww
        return out
    return phi


def exact_density_d2(alpha, beta, N, s, kmax=None, nk=120001):
    """Exact P(d^2 = s) by numerical inversion of P = (1/2pi) int dk e^{-iks} phi(k/N)^N."""
    phi = phi_factory(alpha, beta)
    K = np.linspace(0, kmax or max(3000, 60 * N), nk)
    ph = phi(K / N)**N
    return np.array([np.trapezoid((np.exp(-1j * K * si) * ph).real, K) / np.pi for si in np.atleast_1d(s)])
