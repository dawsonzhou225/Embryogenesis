"""spm.py -- the SPM12 primitives the ADEM step uses, ported one branch each
(spm_speye, spm_DEM_R, spm_DEM_embed, spm_diff, spm_logdet, spm_inv, spm_dx).
spm_dx always builds the augmented matrix sparse and seeds the global generator
around expm_multiply (its 1-norm estimate draws from it), so a run is reproducible
bit for bit.  Same arithmetic as spm_min2.py, the alternatives removed."""
import math
import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv as _inv, cholesky as _chol, svdvals as _svdvals
from scipy.sparse.linalg import expm_multiply

DX = np.exp(-8.0)          # spm_diff's forward-difference step


def spm_speye(m, n=None, k=0):
    """m x n matrix with ones on the k-th diagonal (out-of-range k: all zeros)."""
    if n is None:
        n = m
    D = np.zeros((m, n))
    for i in range(m):
        j = i + k
        if 0 <= j < n:
            D[i, j] = 1.0
    return D


def spm_DEM_R(n, s):
    """Precision of generalised motion to order n, Gaussian autocorrelation, smoothness s."""
    if n == 0:
        return np.zeros((0, 0))
    if s is None:
        return np.zeros((n, n))
    if s == 0:
        s = np.exp(-8.0)
    r = np.zeros(2 * n)
    k = np.arange(n)
    x = np.sqrt(2.0) * s
    r[2 * k] = np.cumprod(np.r_[1.0, 1.0 - 2.0 * np.arange(1, n)]) / (x ** (2.0 * k))
    V = np.zeros((n, n))
    rr = r.copy()
    for i in range(n):
        V[i, :] = rr[i:i + n]
        rr = -rr
    return _inv(V)


def spm_DEM_embed(Y, n, t, dt=1.0):
    """Generalised motion y[0..n-1] of the history Y (q x N) at time t (1-based)."""
    Y = np.atleast_2d(Y)
    q, N = Y.shape
    y = [np.zeros((q, 1)) for _ in range(n)]
    if q == 0:
        return y
    s = float(t) / dt
    k = np.arange(1, n + 1) + int(np.fix(s - (n + 1) / 2.0))
    x = s - k.min() + 1
    k = np.clip(k, 1, N)
    T = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            T[i - 1, j - 1] = ((i - x) * dt) ** (j - 1) / math.factorial(j - 1)
    E = _inv(T)
    Yk = Y[:, k - 1]
    for i in range(n):
        y[i] = Yk @ E[i, :].reshape(-1, 1)
    return y


def spm_diff_vec(f, x0_flat, dx=DX):
    """Forward-difference Jacobian of f and f(x0)."""
    x0_flat = np.asarray(x0_flat, dtype=float).ravel()
    f0 = np.asarray(f(x0_flat), dtype=float).ravel()
    m = x0_flat.size
    J = np.zeros((f0.size, m))
    for i in range(m):
        xi = x0_flat.copy()
        xi[i] += dx
        J[:, i] = (np.asarray(f(xi), dtype=float).ravel() - f0) / dx
    return J, f0


def spm_logdet(C):
    """log determinant, null variances and (near-)null eigenvalues ignored."""
    C = np.asarray(C, dtype=float)
    if C.size == 0:
        return 0.0
    d = np.diag(C)
    i = np.flatnonzero(d)
    if i.size == 0:
        return 0.0
    C = C[np.ix_(i, i)]
    TOL = 1e-16
    if np.allclose(C, np.diag(np.diag(C))):
        s = np.diag(C)
    elif np.linalg.norm(C - C.T, np.inf) > TOL:
        s = _svdvals(C)
    else:
        try:
            R = _chol(C)
            return 2.0 * np.sum(np.log(np.diag(R)))
        except Exception:
            s = _svdvals(C)
    s = np.asarray(s, dtype=float)
    return float(np.sum(np.log(s[(s > TOL) & (s < 1.0 / TOL)])))


def spm_inv(A):
    """inv(A + I*TOL), SPM's tolerance."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if A.size == 0:
        return np.zeros((n, m))
    TOL = max(np.finfo(float).eps * np.linalg.norm(A, np.inf) * max(m, n), np.exp(-32.0))
    return _inv(A + np.eye(m, n) * TOL)


def spm_dx(dfdx, f, t):
    """dx = (expm(t*dfdx) - I) inv(dfdx) f, through the augmented matrix [[0, 0], [t f, t dfdx]]."""
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    A = (dfdx.tocsr() if sp.issparse(dfdx) else sp.csr_matrix(dfdx))
    A = (A * t).tocoo()
    rows = np.concatenate([np.arange(1, n + 1), A.row + 1])
    cols = np.concatenate([np.zeros(n, dtype=int), A.col + 1])
    vals = np.concatenate([t * f, A.data])
    J = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    state = np.random.get_state()
    np.random.seed(0)
    try:
        dx = expm_multiply(J, e0)
    finally:
        np.random.set_state(state)
    return np.real(dx[1:])


def to_csr(A):
    if sp.issparse(A):
        return A.tocsr()
    A = np.ascontiguousarray(A, dtype=float)
    r, c = np.nonzero(A != 0.0)
    return sp.csr_matrix((A[r, c], (r, c)), shape=A.shape)
