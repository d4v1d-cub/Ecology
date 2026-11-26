"""Symbolic derivation/verification (sympy) of the moment formulas used in the notes:
 - E D^2, cumulants kappa_2, kappa_3 of X = D^2 for Beta components
 - identities kappa_2 = 2 mu4 + 2 s^4, kappa_3 = 2 mu6 + 18 mu4 s^2 - 20 mu3^2 - 20 s^6
 - Beta central moments mu3, mu4, mu6 in terms of (mbar, nu) and w = mbar(1-mbar)."""
import sympy as sp
a, b = sp.symbols('alpha beta', positive=True)
def raw(j, a, b): return sp.prod([(a + r) / (a + b + r) for r in range(j)]) if j > 0 else sp.Integer(1)
def ED(n): return sum((-1)**j * sp.binomial(n, j) * raw(j, a, b) * raw(n - j, a, b) for j in range(n + 1))
mu = a / (a + b)
def cm(n, a=a, b=b, mu=mu): return sp.simplify(sum(sp.binomial(n, j) * raw(j, a, b) * (-mu)**(n - j) for j in range(n + 1)))
s2 = a * b / ((a + b)**2 * (a + b + 1))
M2, M4, M6 = ED(2), ED(4), ED(6)
k2 = sp.factor(sp.simplify(M4 - M2**2)); k3 = sp.factor(sp.simplify(M6 - 3 * M4 * M2 + 2 * M2**3))
print("E D^2 - 2 s^2 =", sp.simplify(M2 - 2 * s2))
print("kappa_2 =", k2)
m3, m4, m6 = cm(3), cm(4), cm(6)
print("kappa_2 identity:", sp.simplify(k2 - (2 * m4 + 2 * s2**2)))
print("kappa_3 identity:", sp.simplify(k3 - (2 * m6 + 18 * m4 * s2 - 20 * m3**2 - 20 * s2**3)))
# in terms of p = mbar, nu = alpha + beta
p, nu = sp.symbols('p nu', positive=True)
A, Bb = p * nu, (1 - p) * nu
w = p * (1 - p)
checks = {
    'mu3': (cm(3, A, Bb, p), 2 * w * (1 - 2 * p) / ((nu + 1) * (nu + 2))),
    'mu4': (cm(4, A, Bb, p), 3 * w * (w * (nu - 6) + 2) / ((nu + 1) * (nu + 2) * (nu + 3))),
    'mu6': (cm(6, A, Bb, p), 5 * w * (w**2 * (3 * nu**2 - 86 * nu + 120) + w * (26 * nu - 120) + 24)
            / ((nu + 1) * (nu + 2) * (nu + 3) * (nu + 4) * (nu + 5))),
}
for k, (e, f) in checks.items(): print(k, "(p,nu) form check:", sp.simplify(e - f))
