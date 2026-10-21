"""model.py -- a Markov-blanket organ grown from one cell, and the perturbations that test it.

An organ of 16 ensembles grown from one, every ensemble a blanket of 16 cells grown from one zygote.
Every agent runs the same ADEM step (Palacios et al. 2020, ported in spm.py).  Beyond the paper:
  - a blanket's emission into the organ is its cells' secreted views, one unit per cell / 16
    (a head count); a blanket short of its unit for TAU steps bears one daughter cell
  - the body counts the organ's ensembles the same way; short of a unit, one ensemble divides
    and is born as a zygote of its own, with its fate as a weak prior (e^1)
  - the self-model s: each blanket holds a picture of each of its cells' view of it, pulled
    toward its own self-view (consensus), and its error moves that cell's secreted view
  - daughters are born on a circle at BIRTH_R around their core, in the widest gap of their
    kind; every step each cell (ensemble) is drawn 5% of the way to the radius of its kind
  - the organ's fingerprint: an active ensemble expects 0.11 of active signal, less than the
    paper's 0.276, so a complete ring of blankets does not relabel itself
  - turnover: a complete blanket with too many of one kind sheds its least integrated cell of
    that kind, and the ordinary birth replaces it as the kind the blanket lacks
Perturbations (run's keyword arguments; none by default): cells of blanket 0 removed, its core
removed, whole ensembles removed, cells' kinds flipped or held, a signalling wall through the organ.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, LinAlgError

from spm import spm_speye, spm_DEM_R, spm_DEM_embed, spm_diff_vec, spm_logdet, spm_inv, spm_dx, to_csr

# ---- the paper's constants ---------------------------------------------------------------
E_d, E_n, E_s, E_dt = 1, 2, 1, 2        # DEM: approximation order, embedding order, smoothness, dt
N_TYPE = np.array([1, 6, 9])            # internal, active, sensory in a formed blanket
R_RAD = np.array([0.0, 1.0, 3.0])       # their radii in the paper's template
K_DECAY = 2.0                           # the field kernel exp(-2 d^2)
NG = 16                                 # ensembles the paper's organ is drawn with
# ---- this model's ------------------------------------------------------------------------
Q = 16.0                                # cells per blanket unit; ensembles per body unit
CELL_SCALE = 0.05                       # a blanket's radius in the organ's units
EMIT_FLOOR = 0.25                       # least loudness a cell is normalised by (head count)
THETA = 0.5 / Q                         # deficit that counts (half a cell / ensemble)
TAU, REFRACTORY, SURPLUS = 3, 4, 0.5    # steps short before a birth / turnover; steps between; surplus that counts
V_BIAS = 2.0                            # a determined belief (zygote, daughter's fate)
ORGAN_BIAS = 0.5                        # the organ's seeding belief (the paper's)
POS_BIAS, POS_NOISE, SPREAD = 0.5, 0.125, 0.5   # the organ's seeding positions; the zygote's
PRIOR_PREC, PRIOR_FLOOR = np.exp(4.0), np.exp(-4.0)   # a core's prior; everyone else's
ORGAN_FATE_PREC = np.exp(1.0)           # a daughter ensemble's fate prior, for life
ORGAN_FP_ACT = 0.11                     # an active ensemble expects this much active signal (the paper's template: 0.276)
BIRTH_R = 1.3                           # where a daughter (cell or ensemble) is born, from her core
PULL_R = (1.0, 2.0)                     # the radius of an active, of a sensory
PULL = 0.05                             # fraction of the way to that radius per step
OFFSET, OFFSET_SKIN = 0.3, 1.2          # birth distance from the mother when the ring does not apply
NAMES = ["internal", "active", "sensory"]
KEYS = ("l", "g", "c", "s", "b")        # blocks: cells' kinds, ensembles' kinds, views, pictures, body
PER_CELL = ("l", "c", "s")


def softmax(v):
    e = np.exp(v - v.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def foxhound(x, y, k=K_DECAY):
    """Field sensed at each column of x from the secretions y of the others: (m, n)."""
    d = ((x[:, :, None] - x[:, None, :]) ** 2).sum(axis=0)
    W = np.exp(-k * d)
    np.fill_diagonal(W, 0.0)
    return np.ascontiguousarray(y) @ W.T       # a fixed memory layout: BLAS then sums in the same order every time


def foxhound_wall(x, y, side, k=K_DECAY):
    """foxhound with a signalling wall: nothing is sensed across it (side: which side each column stands on)."""
    d = ((x[:, :, None] - x[:, None, :]) ** 2).sum(axis=0)
    W = np.exp(-k * d) * (side[:, None] == side[None, :])
    np.fill_diagonal(W, 0.0)
    return np.ascontiguousarray(y) @ W.T


def picture(vc):
    """The picture of a view: exp(v.s) = softmax(v.c), volume 1."""
    return vc - np.log(np.exp(vc).sum(axis=0, keepdims=True))


def build_priors():
    """The paper's template: positions, one-hot kinds and the fingerprint P_sen[signal, kind]."""
    pos, typ = [], []
    for j in range(3):
        for i in range(1, N_TYPE[j] + 1):
            th = 2 * np.pi * i / N_TYPE[j]
            pos.append(R_RAD[j] * np.array([np.sin(th), np.cos(th)]))
            typ.append(j)
    pos = np.array(pos).T
    typ = np.array(typ)
    T = np.zeros((3, N_TYPE.sum()))
    T[typ, np.arange(N_TYPE.sum())] = 1.0
    sen = np.zeros_like(T)                    # summed cell by cell, as the original code does
    for i in range(T.shape[1]):
        for j in range(T.shape[1]):
            if i != j:
                dd = pos[:, i] - pos[:, j]
                sen[:, i] += np.exp(-K_DECAY * (dd @ dd)) * T[:, j]
    P = np.zeros((3, 3))
    for i in range(3):
        P[:, i] = sen[:, typ == i].mean(axis=1)
    return pos, T, P


def fate_short(types):
    """The kind furthest below (1, 6, 9) / 16, by fraction of the count."""
    cnt = np.array([(types == k).sum() for k in range(3)], float)
    return int(np.argmax(np.array([1, 6, 9]) / 16.0 - cnt / max(cnt.sum(), 1.0)))


def gap_dir(P, kin, cpos, rng):
    """Unit vector from cpos into the widest angular gap among the columns kin of P."""
    if kin.size >= 2:
        ang = np.sort(np.arctan2(P[1, kin] - cpos[1], P[0, kin] - cpos[0]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        g = int(np.argmax(gaps))
        mid = ang[g] + gaps[g] / 2.0
        u = np.array([np.cos(mid), np.sin(mid)])
    elif kin.size == 1:
        u = cpos - P[:, kin[0]]
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else rng.standard_normal(2)
    else:
        u = rng.standard_normal(2)
    return u / np.linalg.norm(u)


# =============================================================================
# the layout: where each block sits in the action, cause and observation vectors
# =============================================================================
class Layout:
    """N cells, NG ensembles (every one a blanket); own[i] = the ensemble cell i belongs to.
    Per agent: 2 position + 3 secretion actions, 3 causes, 3 secreted + 3 sensed observations."""
    def __init__(self, N, NG_, own):
        self.N, self.NG = N, NG_
        self.own = np.asarray(own, dtype=int)
        self.sizes = dict(l=N, g=NG_, c=N, s=N, b=1)
        self.a_off, self.v_off, self.y_off = {}, {}, {}
        oa = ov = oy = 0
        for k in KEYS:
            n = self.sizes[k]
            self.a_off[k], self.v_off[k], self.y_off[k] = oa, ov, oy
            oa, ov, oy = oa + 5 * n, ov + 3 * n, oy + 6 * n
        self.na, self.nv, self.ny = oa, ov, oy

    def a_unvec(self, a):
        out = {}
        for k in KEYS:
            n, o = self.sizes[k], self.a_off[k]
            out[k] = (a[o:o + 2 * n].reshape(2, n, order="F"), a[o + 2 * n:o + 5 * n].reshape(3, n, order="F"))
        return out

    def a_vec(self, pairs):
        return np.concatenate([x.flatten(order="F") for k in KEYS for x in pairs[k]])

    def v_unvec(self, v):
        return {k: v[self.v_off[k]:self.v_off[k] + 3 * self.sizes[k]].reshape(3, self.sizes[k], order="F") for k in KEYS}

    def v_vec(self, d):
        return np.concatenate([d[k].flatten(order="F") for k in KEYS])

    def y_vec(self, blocks):
        return np.concatenate([x.flatten(order="F") for k in KEYS for x in blocks[k]])

    def hist_unvec(self, A_hist):
        """The action history split by block: {k: (positions (2, n, T), secretions (3, n, T))}."""
        return {k: (A_hist[self.a_off[k]:self.a_off[k] + 2 * self.sizes[k]].reshape(2, self.sizes[k], -1, order="F"),
                    A_hist[self.a_off[k] + 2 * self.sizes[k]:self.a_off[k] + 5 * self.sizes[k]].reshape(3, self.sizes[k], -1, order="F"))
                for k in KEYS}

    def hist_vec(self, Hb, T):
        return np.concatenate([Hb[k][j].reshape((2 if j == 0 else 3) * self.sizes[k], T, order="F") for k in KEYS for j in (0, 1)])


# =============================================================================
# generative process (what the actions produce) and generative model (what the beliefs predict)
# =============================================================================
def Gg(a, L, wall=None):
    """wall: x of a signalling wall in the organ's frame, or None."""
    A = L.a_unvec(a)
    N, NG_, own = L.N, L.NG, L.own
    l_pos, l_sec = A["l"]
    g_pos = A["g"][0]                       # an ensemble's own secretion action is never read: it is heard through its cells
    c_pos, c_sec = A["c"]
    w = 1.0 / Q
    src = c_sec / np.maximum(c_sec.sum(axis=0, keepdims=True), EMIT_FLOOR)   # each cell one unit: a head count
    cg = g_pos[:, own] + CELL_SCALE * c_pos                                  # the cells in the organ's frame
    side = None if wall is None else (cg[0] >= wall)
    field = (lambda x, y, m: foxhound(x, y)) if wall is None else (lambda x, y, m: foxhound_wall(x, y, side[m]))
    l_sen = np.zeros_like(l_sec)                                             # each blanket's local field: its own cells
    for k in range(NG_):
        idx = np.flatnonzero(own == k)
        l_sen[:, idx] = field(l_pos[:, idx], l_sec[:, idx], idx)
    c_sen = field(cg, src * w, slice(None))                                  # the organ's field, from every cell
    g_sec = np.zeros((3, NG_))
    g_sen = np.zeros((3, NG_))
    for k in range(NG_):
        idx = np.flatnonzero(own == k)
        c_sen[:, idx] = c_sen[:, idx] - field(cg[:, idx], src[:, idx] * w, idx)   # a view senses the OTHER blankets
        g_sec[:, k] = src[:, idx].sum(axis=1) * w                            # the blanket IS its cells' views, N_k / 16
        g_sen[:, k] = c_sen[:, idx].mean(axis=1)
    bsec = (g_sec / np.maximum(g_sec.sum(axis=0, keepdims=True), 1e-9)).sum(axis=1, keepdims=True) / Q   # the body counts heads
    return L.y_vec({"l": (l_sec, l_sen), "g": (g_sec, g_sen), "c": (c_sec, c_sen),
                    "s": (c_sec, np.zeros((3, N))), "b": (bsec, np.zeros((3, 1)))})


def Mg(v, L, P_sen, P_org):
    V = L.v_unvec(v)
    pl, pg, pc = softmax(V["l"]), softmax(V["g"]), softmax(V["c"])
    pG = pg[:, L.own]                                   # each cell's own ensemble's self-view
    ex = np.exp(V["s"])                                 # the picture, with its loudness
    vol = ex.sum(axis=0, keepdims=True)
    p = softmax(V["s"])
    return L.y_vec({"l": (pl, P_sen @ pl), "g": (pg, P_org @ pg), "c": (pc, P_org @ pc),
                    "s": (ex, vol * (p - pG)),          # s_sec: observed as c_sec; s_sen: consensus, observed as 0
                    "b": (softmax(V["b"]), np.zeros((3, 1)))})


# =============================================================================
# the fixed matrices of the ADEM step for a layout
# =============================================================================
def restriction(L):
    """Which observation's error each action answers: an agent's own block, plus
    up: a blanket's g rows answer to its cells' view actuators (the paper's credit);
    down: the s_sec row about cell i answers to cell i's view actuators (the self-model's)."""
    R = np.zeros((L.ny, L.na))
    for k in KEYS:
        n = L.sizes[k]
        I = np.eye(n)
        R1 = np.hstack([np.kron(I, np.ones((3, 2))), np.kron(I, np.ones((3, 3)))])
        R1 = np.vstack([R1, R1])
        R[L.y_off[k]:L.y_off[k] + 6 * n, L.a_off[k]:L.a_off[k] + 5 * n] = R1
    gy, oc, N, NG_ = L.y_off["g"], L.a_off["c"], L.N, L.NG
    for k in range(NG_):
        g_obs = np.r_[gy + 3 * k:gy + 3 * k + 3, gy + 3 * NG_ + 3 * k:gy + 3 * NG_ + 3 * k + 3]
        cells = np.flatnonzero(L.own == k)
        ck_act = np.concatenate([oc + 2 * cells, oc + 2 * cells + 1,
                                 oc + 2 * N + 3 * cells[:, None] + np.arange(3)[None, :]], axis=None)
        R[np.ix_(g_obs, ck_act)] = 1.0
    s_obs = L.y_off["s"]
    for i in range(N):
        ci = np.r_[oc + 2 * i:oc + 2 * i + 2, oc + 2 * N + 3 * i:oc + 2 * N + 3 * i + 3]
        R[np.ix_(np.arange(s_obs + 3 * i, s_obs + 3 * i + 3), ci)] = 1.0
    return R


class Fixed:
    def __init__(self, L, pprec_l, g_pprec):
        n, d = E_n + 1, E_d + 1
        self.L, self.n, self.d = L, n, d
        nv, ny, na = L.nv, L.ny, L.na
        nc = nv
        ne = ny + nc
        gy = ny
        gr = gy + 1
        self.nv, self.ny, self.na, self.nc, self.ne, self.gy, self.gr = nv, ny, na, nc, ne, gy, gr
        iV = spm_DEM_R(n, E_s)
        pc = np.full(nc, PRIOR_FLOOR)                                   # priors: the floor everywhere,
        pc[L.v_off["l"]:L.v_off["l"] + 3 * L.N] = np.repeat(np.asarray(pprec_l, float), 3)   # the cells' (cores e^4),
        pc[L.v_off["g"]:L.v_off["g"] + 3 * L.NG] = np.repeat(np.asarray(g_pprec, float), 3)  # the ensembles' fates (e^1),
        pc[L.v_off["g"]:L.v_off["g"] + 3] = PRIOR_PREC                  # the organ's core (e^4)
        mv = np.r_[np.full(ny, np.exp(4.0)), pc]                       # observations at e^4
        gu = np.r_[np.full(gy, np.exp(-2.0)), np.zeros(ne - gy)]       # action weighs errors at e^-2
        Rr = np.zeros((gr, na))
        Rr[:ny, :] = restriction(L)
        self.Pu = np.kron(iV[:d, :d], np.eye(nv) * np.exp(-8.0))
        self.Pa = np.eye(na) * np.exp(-8.0)
        self.dedy = spm_speye(ne, ny)
        self.dedc = -spm_speye(ne, nc, nc - ne)
        sk = lambda A, B: sp.kron(to_csr(A), to_csr(B), format="csr")
        self.iS_s = sk(iV, sp.diags(mv))
        self.iS_T = self.iS_s.T.tocsr()
        self.iG_T = sk(iV, sp.diags(gu)).T.tocsr()
        self.R_s = sk(spm_speye(n, 1, 0), Rr)
        Dv_m = sk(spm_speye(d, d, 1), sp.identity(nv))
        Dc = sk(spm_speye(d, d, 1), sp.identity(nc))
        Da = sp.csr_matrix((na, na))
        self.Du_s, self.Dc_s, self.Da_s = Dv_m, Dc, Da
        self.Dq_s = sp.block_diag([Dv_m, Dc, Da], format="csr")
        self.Dv_p_s = sk(spm_speye(n, n, 1), sp.identity(gr))
        self.Dp_s = sp.block_diag([self.Dv_p_s, self.Dv_p_s], format="csr")
        dydv = sk(np.eye(n), spm_speye(gy, gr))
        self.dE_dv_s = (sk(spm_speye(n, n), self.dedy) @ dydv).tocsr()
        self.dE_dc_s = sk(spm_speye(n, d), self.dedc)
        self.Pu_s, self.Pa_s = to_csr(self.Pu), to_csr(self.Pa)
        self.speye_n1 = to_csr(spm_speye(n, 1, 0))
        if n * ne <= 8000:
            self.logdet_iS = spm_logdet(np.kron(iV, np.diag(mv)))
        else:
            self.logdet_iS = ne * spm_logdet(iV) + n * float(np.sum(np.log(mv)))
        self.o_quv = 2 * n * gr
        self.o_quu = self.o_quv + d * nv
        self.o_qua = self.o_quu + d * nc
        self.n_tot = self.o_qua + na


# =============================================================================
# one ADEM step: every agent perceives and acts at once
# =============================================================================
def adem_step(fx, Gfun, Mfun, v, vd, a, A_hist, iY, prior_mean, dt=E_dt):
    """v, vd: causes and their motion; a: actions; A_hist: (na, T).  Returns the new v, vd, a,
    this step's observations y, and the free-energy terms."""
    n, d = fx.n, fx.d
    nv, nc, ne, gr, gy, na = fx.nv, fx.nc, fx.ne, fx.gr, fx.gy, fx.na
    qu_v = [v, vd]
    pu_a = [x.ravel() for x in spm_DEM_embed(A_hist, n, iY)]
    pu_z = [np.zeros(gr) for _ in range(n)]
    qu_u = [np.zeros(nc) for _ in range(n)]
    qu_u[0] = prior_mean
    dgda_1, g1 = spm_diff_vec(Gfun, pu_a[0])                   # the process and its slope
    dg_da = np.zeros((gr, na))
    dg_da[:gy, :] = dgda_1
    pu_v = [np.concatenate([g1, np.zeros(1)]) + pu_z[0]]
    for i in range(1, n):
        pu_v.append(dg_da @ pu_a[i] + pu_z[i])
    qu_y = [pu_v[i][:gy] for i in range(n)]
    dgdv_1, g_pred = spm_diff_vec(Mfun, qu_v[0])               # the model and its slope
    dgdv = np.vstack([dgdv_1, -np.eye(nv)])
    E = np.zeros(n * ne)                                        # prediction errors: observations, then priors
    E[:ne] = np.concatenate([qu_y[0], qu_v[0]]) - np.concatenate([g_pred, qu_u[0]])
    for i in range(1, n):
        vi = qu_v[i] if i < d else np.zeros(nv)
        E[i * ne:(i + 1) * ne] = fx.dedy @ qu_y[i] + fx.dedc @ qu_u[i] - dgdv @ vi
    dE_du = to_csr(-np.kron(spm_speye(n, d), dgdv))
    dgda_gen = sp.kron(fx.speye_n1, to_csr(dg_da), format="csr")
    dE_da = fx.dE_dv_s @ dgda_gen.multiply(fx.R_s)             # the restriction: who answers which error
    A = (fx.iS_T @ dE_du).T.tocsr()
    B = (fx.iG_T @ dE_da).T.tocsr()
    AdEdu = (A @ dE_du).toarray()
    C = AdEdu + fx.Pu
    try:
        Cs = 0.5 * (C + C.T)
        tol = max(np.finfo(float).eps * np.linalg.norm(Cs, np.inf) * Cs.shape[0], np.exp(-32.0))
        Lc = cho_factor(Cs + np.eye(Cs.shape[0]) * tol, lower=True, check_finite=False)
        ld_quc = -2.0 * float(np.sum(np.log(np.diag(Lc[0]))))
    except (LinAlgError, ValueError):
        ld_quc = spm_logdet(spm_inv(C))
    u_states = np.concatenate(qu_v[:d])
    dVdu = -A @ E - fx.Pu_s @ u_states                          # gradients on beliefs and actions
    dVda = -B @ E - fx.Pa_s @ a
    dVdc = np.zeros(d * nc)
    dVduu = -AdEdu - fx.Pu
    dVdaa = -B @ dE_da - fx.Pa_s
    dVduv = -A @ fx.dE_dv_s
    dVduc = -A @ fx.dE_dc_s
    dVdua = -A @ dE_da
    dVdav = -B @ fx.dE_dv_s
    dVdau = -B @ dE_du
    dVdac = -B @ fx.dE_dc_s
    u = np.zeros(fx.n_tot)
    u[:n * gr] = np.concatenate(pu_v)
    u[n * gr:2 * n * gr] = np.concatenate(pu_z)
    u[fx.o_quv:fx.o_quv + d * nv] = np.concatenate(qu_v[:d])
    u[fx.o_quu:fx.o_quu + d * nc] = np.concatenate(qu_u[:d])
    u[fx.o_qua:] = a
    dFdu = np.zeros(fx.n_tot)
    dFdu[:2 * n * gr] = fx.Dp_s @ u[:2 * n * gr]
    dFdu[fx.o_quv:] = np.concatenate([dVdu, dVdc, dVda]) + fx.Dq_s @ u[fx.o_quv:]
    dFduu = sp.bmat([
        [None,  fx.Dv_p_s, None,                       None,     None],
        [None,  fx.Dv_p_s, None,                       None,     None],
        [dVduv, None,      fx.Du_s + to_csr(dVduu),    dVduc,    dVdua],
        [None,  None,      None,                       fx.Dc_s,  None],
        [dVdav, None,      dVdau,                      dVdac,    dVdaa + fx.Da_s]], format="csr")
    if not (np.all(np.isfinite(dFdu)) and np.all(np.isfinite(dFduu.data))):
        raise FloatingPointError("non-finite gradient or Jacobian")
    du = None
    for k in range(7):                                          # the local-linearisation step, halved on failure
        try:
            cand = spm_dx(dFduu, dFdu, dt / 2 ** k)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.all(np.isfinite(cand)):
            du = cand
            break
    if du is None:
        raise FloatingPointError("integration step not finite")
    u = u + du
    v_new = u[fx.o_quv:fx.o_quv + nv]
    vd_new = u[fx.o_quv + nv:fx.o_quv + 2 * nv]
    a_new = u[fx.o_qua:]
    EiSE_t = float(E @ (fx.iS_s @ E))
    Jt = -0.5 * EiSE_t + ld_quc + 0.5 * fx.logdet_iS
    return v_new, vd_new, a_new, qu_y[0], Jt, EiSE_t, ld_quc


# =============================================================================
# the run
# =============================================================================
def run(seed=0, nY=256, ablate=None, organ_ablate=None, mutate=None, wall=None):
    """Grow the organ for nY steps.  Perturbations, all on blanket 0 (the organ's core):
      ablate = (step, what, n)      remove n cells of kind what ("sensory" | "active" | "random"), or ("core")
      organ_ablate = (step, n)      remove n whole ensembles (never the organ's core), cells and all
      mutate = (step, n, how)       n ring cells flip kind; how = "flip" (belief only) | "hold" (a prior for life)
      wall = (step, until)          a signalling wall through the organ's core from step to until"""
    rng = np.random.default_rng(seed)
    pos0, typ, P_sen = build_priors()
    P_org = P_sen.copy()
    P_org[1, 1] = ORGAN_FP_ACT

    # the seed: the organ drawn as the paper draws it, its core kept; the zygote, a determined internal cell
    v0 = typ * ORGAN_BIAS + rng.standard_normal(typ.shape) / 8.0
    POSg = pos0 * POS_BIAS + rng.standard_normal(pos0.shape) * POS_NOISE
    Vg, POSg = v0[:, :1].copy(), POSg[:, :1].copy()
    Vl0 = np.zeros((3, 1))
    Vl0[0, :] = V_BIAS
    Vl = Vl0 + rng.standard_normal(Vl0.shape) / 8.0
    POSl = rng.standard_normal((2, 1)) * SPREAD
    Vc = np.tile(typ[:, [0]], (1, 1)) * V_BIAS + np.random.default_rng([seed, 7, 0]).standard_normal((3, 1)) / 8.0
    N = 1
    own = np.zeros(N, dtype=int)
    L = Layout(N, Vg.shape[1], own)
    Vs = picture(Vc)
    V = L.v_vec(dict(l=Vl, g=Vg, c=Vc, s=Vs, b=np.zeros((3, 1))))
    Vd = np.zeros_like(V)
    a = L.a_vec(dict(l=(POSl, softmax(Vl)), g=(POSg, softmax(Vg)), c=(POSl.copy(), softmax(Vc)),
                     s=(POSl.copy(), softmax(Vs)), b=(np.zeros((2, 1)), softmax(np.zeros((3, 1))))))
    A_hist = a[:, None].copy()

    prior_mean_l = Vl0.copy()                           # the cells' priors: the cores internal at e^4, the rest none
    pprec = np.full(N, PRIOR_PREC)
    g_pmean = np.zeros((3, L.NG))                       # the ensembles' priors: each daughter's fate at e^1
    g_pprec = np.full(L.NG, PRIOR_FLOOR)
    g0_mean = np.array([V_BIAS, 0.0, 0.0])              # the organ's core: internal at e^4
    born = np.full(N, -10 ** 6)                         # per cell: birth step, last division
    last_div = np.full(N, -10 ** 6)
    anchor = {0: 0}                                     # per blanket: its core, steps short, last birth
    streak_p, streak_k = {0: 0}, {0: 0}                 # ... steps with a surplus of one kind, last turnover
    last_parent_div, last_prune = {0: -10 ** 6}, {0: -10 ** 6}
    streak_o, last_organ_div = 0, -10 ** 6              # the organ's
    wall_x = [None]                                     # the wall's x while it stands

    def make_fixed():
        return Fixed(L, pprec, g_pprec)

    def prior_vec():
        z = np.zeros((3, N))
        pg_ = g_pmean.copy()
        pg_[:, 0] = g0_mean
        return L.v_vec(dict(l=prior_mean_l, g=pg_, c=z, s=z, b=np.zeros((3, 1))))

    def rebuild(Vb, Vdb, Ab, Hb, T):
        nonlocal L, V, Vd, a, A_hist, Gf, Mf, fx
        L = Layout(N, Vb["g"].shape[1], own)
        V, Vd, a = L.v_vec(Vb), L.v_vec(Vdb), L.a_vec(Ab)
        A_hist = L.hist_vec(Hb, T)
        Gf = lambda aa, L=L: Gg(aa, L, wall_x[0])
        Mf = lambda vv, L=L: Mg(vv, L, P_sen, P_org)
        fx = make_fixed()

    def add_cell(Vb, Vdb, Ab, Hb, T, newv, vd_c, i, dpos):
        """One cell's columns in the per-cell blocks: beliefs newv (l, c; s = its picture), the
        c and s motion vd_c.  A daughter (i, dpos) stands dpos from cell i in every block, her
        position history cell i's shifted; a zygote (None) at the origin.  Secretions repeated."""
        newv = dict(newv, s=picture(newv["c"]))
        for k in PER_CELL:
            sec = softmax(newv[k])
            Vb[k] = np.hstack([Vb[k], newv[k]])
            Vdb[k] = np.hstack([Vdb[k], np.zeros((3, 1)) if k == "l" else vd_c[k]])
            P_, S_ = Ab[k]
            Hp, Hs = Hb[k]
            if i is None:
                pos, hp = np.zeros((2, 1)), np.zeros((2, 1, T))
            else:
                pos, hp = P_[:, [i]] + dpos[:, None], Hp[:, [i], :] + dpos[None, :, None].transpose(1, 0, 2)
            Ab[k] = (np.hstack([P_, pos]), np.hstack([S_, sec]))
            Hb[k] = (np.concatenate([Hp, hp], axis=1), np.concatenate([Hs, np.repeat(sec[:, :, None], T, axis=2)], axis=1))

    Gf = lambda aa: Gg(aa, L, wall_x[0])
    Mf = lambda vv: Mg(vv, L, P_sen, P_org)
    fx = make_fixed()
    rec = dict(N=[], types=[], conf=[], pos=[], deficit=[], sen=[], p=[], own=[], deficit_by=[],
               organ_types=[], organ_N=[], organ_pos=[], organ_emit=[], views=[], organ_sec=[], organ_sen=[],
               self_cons_by=[], J=[], divisions=[], celled_births=[], organ_divisions=[], removals=[],
               organ_removals=[], mutations=[], wall=[], err_by=[], reanchor=[])
    EiSE = Hquc = 0.0
    t0 = time.time()
    for iY in range(1, nY + 1):
        # 0. the perturbations that act before the step: the wall up / down, the mutation
        if wall and iY == wall[0]:
            wall_x[0] = float(a[L.a_off["g"]])                               # through the organ's core
            rec["wall"].append((iY, wall_x[0], "up"))
        if wall and iY == wall[1] and wall_x[0] is not None:
            rec["wall"].append((iY, wall_x[0], "down"))
            wall_x[0] = None
        if mutate and iY == mutate[0]:
            ty_m = softmax(L.v_unvec(V)["l"]).argmax(axis=0)
            pool = np.flatnonzero((own == 0) & (ty_m > 0))
            pool = pool[pool != anchor[0]]
            picked = rng.choice(pool, min(mutate[1], pool.size), replace=False)
            ol = L.v_off["l"]
            for i in picked:
                i, new_k = int(i), 3 - int(ty_m[i])                          # active <-> sensory
                V[ol + 3 * i:ol + 3 * i + 3] = 0.0
                V[ol + 3 * i + new_k] = V_BIAS
                Vd[ol + 3 * i:ol + 3 * i + 3] = 0.0
                if mutate[2] == "hold":                                      # a prior it cannot shed
                    prior_mean_l[:, i] = 0.0
                    prior_mean_l[new_k, i] = V_BIAS
                    pprec[i] = PRIOR_PREC
                rec["mutations"].append((iY, i, int(ty_m[i]), new_k, mutate[2]))
            if mutate[2] == "hold" and picked.size:
                fx = make_fixed()
        # 1. the ADEM step
        V, Vd, a, y, Jt, e2, ld = adem_step(fx, Gf, Mf, V, Vd, a, A_hist, iY, prior_vec())
        EiSE += e2
        Hquc += ld
        err_ = y - Mf(V)                                                     # the residual error by block, for the figures
        err_by = {}
        for k in KEYS:
            o, n_ = L.y_off[k], L.sizes[k]
            err_by[k + "_sec"] = round(float((err_[o:o + 3 * n_] ** 2).sum()), 6)
            err_by[k + "_sen"] = round(float((err_[o + 3 * n_:o + 6 * n_] ** 2).sum()), 6)
        rec["err_by"].append(err_by)
        # 2. the cores held where they stood: each blanket's in l, c, s; the organ's in g
        for j in sorted(j for j in anchor.values() if j is not None):
            for k in PER_CELL:
                o = L.a_off[k] + 2 * j
                a[o:o + 2] = A_hist[o:o + 2, -1]
        o = L.a_off["g"]
        a[o:o + 2] = A_hist[o:o + 2, -1]
        # 3. the cells' pull: each non-core, non-internal cell drawn PULL of the way to its kind's radius
        t_now = softmax(L.v_unvec(V)["l"]).argmax(axis=0)
        ol = L.a_off["l"]
        Pl_now = a[ol:ol + 2 * N].reshape(2, N, order="F")
        for k_ in range(L.NG):
            idx = np.flatnonzero(own == k_)
            ci = anchor[k_]
            if ci is None:                                                   # no core: the nearest core-believer, else the centroid
                cen = Pl_now[:, idx].mean(axis=1)
                cores_ = idx[t_now[idx] == 0]
                ci = int(cores_[np.argmin(np.linalg.norm(Pl_now[:, cores_] - cen[:, None], axis=0))]) if cores_.size else -1
            cpos = Pl_now[:, ci] if ci >= 0 else Pl_now[:, idx].mean(axis=1)
            for j in idx:
                if j == ci or t_now[j] == 0:
                    continue
                v_ = Pl_now[:, j] - cpos
                d_ = float(np.linalg.norm(v_))
                if d_ < 1e-6:
                    continue
                delta = PULL * (cpos + float(PULL_R[t_now[j] - 1]) * v_ / d_ - Pl_now[:, j])
                for k in PER_CELL:
                    o = L.a_off[k] + 2 * j
                    a[o:o + 2] += delta
        # 4. the ensembles' pull, around the organ's core
        gt_now = softmax(L.v_unvec(V)["g"]).argmax(axis=0)
        og = L.a_off["g"]
        Pg_now = a[og:og + 2 * L.NG].reshape(2, L.NG, order="F")
        cpos = Pg_now[:, 0]
        for k_ in range(L.NG):
            if k_ == 0 or gt_now[k_] == 0:
                continue
            v_ = Pg_now[:, k_] - cpos
            d_ = float(np.linalg.norm(v_))
            if d_ < 1e-6:
                continue
            o = og + 2 * k_
            a[o:o + 2] += PULL * (cpos + float(PULL_R[gt_now[k_] - 1]) * v_ / d_ - Pg_now[:, k_])
        # 5. the corrected actions join the history
        A_hist = np.hstack([A_hist, a[:, None]])

        # 6. read the step off and record it
        Vb, Ab = L.v_unvec(V), L.a_unvec(a)
        p = softmax(Vb["l"])
        types = p.argmax(axis=0)
        sen = y[L.y_off["l"] + 3 * N:L.y_off["l"] + 6 * N].reshape(3, N, order="F")
        pred = P_sen[:, types]                                               # each cell's local deficit
        expected = pred.sum(axis=0)
        cdef = np.clip(pred - sen, 0.0, None).sum(axis=0)
        cdef = np.where(expected > 1e-9, cdef / np.maximum(expected, 1e-9), 0.0)
        g_sec_all = y[L.y_off["g"]:L.y_off["g"] + 3 * L.NG].reshape(3, L.NG, order="F")
        g_sen_all = y[L.y_off["g"] + 3 * L.NG:L.y_off["g"] + 6 * L.NG].reshape(3, L.NG, order="F")
        pdef_by = {k_: 1.0 - float(g_sec_all[:, k_].sum()) for k_ in range(L.NG)}   # each blanket's deficit: 1 - N_k / 16
        pg_all = softmax(Vb["g"])
        ps = softmax(Vb["s"])
        pG_own = pg_all[:, own]
        organ_emit = float(y[L.y_off["b"]:L.y_off["b"] + 3].sum())             # the body's count: NG / 16
        for k_ in sorted(anchor):                                               # a blanket without a core adopts a new one
            if anchor[k_] is None and ((types == 0) & (own == k_)).any():
                anchor[k_] = int(np.flatnonzero((types == 0) & (own == k_))[0])
                rec["reanchor"].append(iY)
        rec["N"].append(int(N))
        rec["types"].append(types.tolist())
        rec["conf"].append(np.round(p.max(axis=0), 4).tolist())
        rec["pos"].append(np.round(Ab["l"][0], 4).tolist())
        rec["deficit"].append(np.round(cdef, 4).tolist())
        rec["sen"].append(np.round(sen, 4).tolist())
        rec["p"].append(np.round(p, 4).tolist())
        rec["own"].append(own.tolist())
        rec["deficit_by"].append({int(k_): round(v_, 4) for k_, v_ in pdef_by.items()})
        rec["organ_types"].append(pg_all.argmax(axis=0).tolist())
        rec["organ_N"].append(int(L.NG))
        rec["organ_pos"].append(np.round(Ab["g"][0], 3).tolist())
        rec["organ_emit"].append(round(organ_emit, 4))
        rec["views"].append(softmax(Vb["c"]).argmax(axis=0).tolist())
        rec["organ_sec"].append(np.round(g_sec_all, 4).tolist())
        rec["organ_sen"].append(np.round(g_sen_all, 4).tolist())
        rec["self_cons_by"].append({int(k_): round(float(np.abs(ps[:, own == k_] - pG_own[:, own == k_]).sum(axis=0).mean()), 4)
                                    for k_ in range(L.NG)})
        rec["J"].append(float(Jt))

        # 7. blanket births: a blanket short for TAU steps bears one daughter
        new = []
        eligible = (iY - last_div >= REFRACTORY) & (iY - born >= REFRACTORY)
        for k_ in range(L.NG):
            streak_p[k_] = streak_p[k_] + 1 if pdef_by[k_] > THETA else 0
            elig_k = eligible & (own == k_)
            if streak_p[k_] >= TAU and iY - last_parent_div[k_] >= REFRACTORY and elig_k.any():
                cand = np.flatnonzero(elig_k)
                fate = fate_short(types[own == k_])                          # the kind the blanket is short of
                inner = cand[types[cand] == max(fate - 1, 0)]                # the mother: the layer inside it,
                pool = inner if inner.size else cand                         # its most deficient cell
                new.append(int(pool[np.argmax(cdef[pool] + 1e-9 * rng.random(pool.size))]))
                last_parent_div[k_] = iY
                streak_p[k_] = 0
        if new:
            Vb_ = {k: Vb[k].copy() for k in KEYS}
            Vdb = L.v_unvec(Vd)
            Vdb_ = {k: Vdb[k].copy() for k in KEYS}
            Ab_ = {k: (Ab[k][0].copy(), Ab[k][1].copy()) for k in KEYS}
            Hn = {k: (h[0].copy(), h[1].copy()) for k, h in L.hist_unvec(A_hist).items()}
            T = A_hist.shape[1]
            Pl = Ab_["l"][0]
            for i in new:
                sib = np.flatnonzero(own[:N] == own[i])
                fk = fate_short(types[sib])
                if sib.size > 1 and fk in (1, 2):
                    # on the circle at BIRTH_R around the core, in the widest gap of her kind
                    cen = Pl[:, sib].mean(axis=1)
                    cores = sib[types[sib] == 0]
                    cpos = Pl[:, cores[np.argmin(np.linalg.norm(Pl[:, cores] - cen[:, None], axis=0))]] if cores.size else cen
                    u = gap_dir(Pl, sib[types[sib] == fk], cpos, rng)
                    dpos = cpos + BIRTH_R * u - Pl[:, i]
                else:                                                        # the zygote's first daughter, or an internal one
                    u = rng.standard_normal(2)
                    u = u / np.linalg.norm(u)
                    dpos = (OFFSET_SKIN if fk == 2 else OFFSET) * u
                vl_new = np.zeros((3, 1))
                vl_new[fk, 0] = V_BIAS                                       # she believes her fate
                vl_new = vl_new + rng.standard_normal((3, 1)) / 8.0
                vc_new = Vb_["c"][:, [i]] + rng.standard_normal((3, 1)) / 8.0   # and inherits her mother's view
                add_cell(Vb_, Vdb_, Ab_, Hn, T, {"l": vl_new, "c": vc_new}, {k: Vdb_[k][:, [i]] for k in ("c", "s")}, i, dpos)
                born = np.append(born, iY)
                last_div = np.append(last_div, iY)
                last_div[i] = iY
                own = np.append(own, own[i])
                prior_mean_l = np.hstack([prior_mean_l, np.zeros((3, 1))])  # no prior: her kind is read off the field
                pprec = np.append(pprec, PRIOR_FLOOR)
                rec["divisions"].append((iY, int(i), int(Vb_["l"].shape[1] - 1), int(types[i])))
            N = Vb_["l"].shape[1]
            rebuild(Vb_, Vdb_, Ab_, Hn, T)

        # 7a. ablation: cells of blanket 0 removed at a step (turnover stands down that step)
        gone = []
        if ablate and iY == ablate[0]:
            if ablate[1] == "core":
                gone = [anchor[0]] if anchor[0] is not None else []
            else:
                pool = np.flatnonzero(own[:types.size] == 0)
                if ablate[1] in ("active", "sensory"):
                    pool = pool[types[pool] == (1 if ablate[1] == "active" else 2)]
                pool = pool[pool != anchor[0]]
                gone = [int(j) for j in rng.choice(pool, min(ablate[2], pool.size), replace=False)]
            for j in gone:
                rec["removals"].append((iY, j, int(types[j]), "ablate"))
        # 7b. turnover: a complete blanket with too many of one kind sheds its least integrated
        #     cell of that kind (never its core); the next birth is of the kind it lacks
        for k_ in (range(L.NG) if not gone else []):
            cells_k = np.flatnonzero(own[:types.size] == k_)                 # cells that stood at the step's start
            if cells_k.size <= 1:
                continue
            ty_k = types[cells_k]
            surplus = np.array([(ty_k == k).sum() for k in range(3)], float) - N_TYPE
            streak_k[k_] = streak_k[k_] + 1 if (pdef_by[k_] <= THETA and surplus.max() > SURPLUS) else 0
            if streak_k[k_] >= TAU and iY - last_prune[k_] >= REFRACTORY:
                pool = cells_k[ty_k == int(np.argmax(surplus))]
                pool = pool[pool != anchor[k_]]
                if pool.size:
                    j = int(pool[np.argmax(cdef[pool] + 1e-9 * rng.random(pool.size))])
                    gone.append(j)
                    rec["removals"].append((iY, j, int(types[j]), "turnover"))
                    last_prune[k_] = iY
                streak_k[k_] = 0
        if gone:
            keep = np.array([j for j in range(N) if j not in gone])
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            for k in PER_CELL:
                Vb[k], Vdb[k] = Vb[k][:, keep], Vdb[k][:, keep]
                Ab[k] = (Ab[k][0][:, keep], Ab[k][1][:, keep])
                Hb[k] = (Hb[k][0][:, keep, :], Hb[k][1][:, keep, :])
            born, last_div, pprec = born[keep], last_div[keep], pprec[keep]
            prior_mean_l = prior_mean_l[:, keep]
            for k_ in anchor:
                anchor[k_] = None if anchor[k_] is None or anchor[k_] not in keep else int(np.flatnonzero(keep == anchor[k_])[0])
            own = own[keep]
            N = keep.size
            rebuild(Vb, Vdb, Ab, Hb, T)
            for k_ in streak_p:                                              # every blanket counts afresh
                streak_p[k_] = 0

        # 7c. whole ensembles removed at a step, their cells with them (never the organ's core)
        organ_changed = False
        if organ_ablate and iY == organ_ablate[0]:
            ggone = [int(j) for j in rng.choice(np.arange(1, L.NG), min(organ_ablate[1], L.NG - 1), replace=False)]
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            gtypes = softmax(Vb["g"]).argmax(axis=0)
            for j in ggone:
                rec["organ_removals"].append((iY, j, int(gtypes[j]), "ablate"))
            keep = np.array([j for j in range(L.NG) if j not in ggone])
            remap = {int(old): new for new, old in enumerate(keep.tolist())}
            Vb["g"], Vdb["g"] = Vb["g"][:, keep], Vdb["g"][:, keep]
            Ab["g"] = (Ab["g"][0][:, keep], Ab["g"][1][:, keep])
            Hb["g"] = (Hb["g"][0][:, keep, :], Hb["g"][1][:, keep, :])
            g_pmean, g_pprec = g_pmean[:, keep], g_pprec[keep]
            ckeep = np.flatnonzero(np.isin(own, keep))
            for k in PER_CELL:
                Vb[k], Vdb[k] = Vb[k][:, ckeep], Vdb[k][:, ckeep]
                Ab[k] = (Ab[k][0][:, ckeep], Ab[k][1][:, ckeep])
                Hb[k] = (Hb[k][0][:, ckeep, :], Hb[k][1][:, ckeep, :])
            born, last_div, pprec = born[ckeep], last_div[ckeep], pprec[ckeep]
            prior_mean_l = prior_mean_l[:, ckeep]
            for d_ in (anchor, streak_p, last_parent_div, streak_k, last_prune):
                for kk in list(d_):
                    if kk not in remap:
                        del d_[kk]
            for kk in anchor:
                anchor[kk] = int(np.flatnonzero(ckeep == anchor[kk])[0]) if anchor[kk] is not None else None
            for d_ in (anchor, streak_p, last_parent_div, streak_k, last_prune):
                for kk in sorted(d_):
                    if remap[kk] != kk:
                        d_[remap[kk]] = d_.pop(kk)
            own = np.array([remap[int(o)] for o in own[ckeep]], dtype=int)
            N = ckeep.size
            rebuild(Vb, Vdb, Ab, Hb, T)
            streak_o, organ_changed = 0, True

        # 8. the organ's birth: the body short for TAU steps -> one ensemble divides, born as a zygote of its own
        streak_o = 0 if organ_changed else (streak_o + 1 if 1.0 - organ_emit > THETA else 0)
        if streak_o >= TAU and iY - last_organ_div >= REFRACTORY:
            Vb, Vdb, Ab = L.v_unvec(V), L.v_unvec(Vd), L.a_unvec(a)
            Hb = L.hist_unvec(A_hist)
            T = A_hist.shape[1]
            Pg, Sg = Ab["g"]
            gtypes = softmax(Vb["g"]).argmax(axis=0)
            fate_g = fate_short(gtypes)
            inner = np.flatnonzero(gtypes == max(fate_g - 1, 0))
            pool = inner if inner.size else np.arange(L.NG)
            im = int(rng.choice(pool)) if pool.size > 1 else int(pool[0])
            if fate_g in (1, 2):
                cen = Pg.mean(axis=1)
                gcores = np.flatnonzero(gtypes == 0)
                cpos = Pg[:, 0] if gtypes[0] == 0 else (Pg[:, gcores[np.argmin(np.linalg.norm(Pg[:, gcores] - cen[:, None], axis=0))]] if gcores.size else cen)
                u = gap_dir(Pg, np.flatnonzero(gtypes == fate_g), cpos, rng)
                dpos = cpos + BIRTH_R * u - Pg[:, im]
            else:                                                            # an internal daughter: into the gap of the mother's field
                d = Pg - Pg[:, [im]]
                w = np.exp(-K_DECAY * (d ** 2).sum(axis=0)) * Sg.sum(axis=0)
                w[im] = 0.0
                gvec = (d * w[None, :]).sum(axis=1)
                u = -gvec if np.linalg.norm(gvec) > 1e-6 else rng.standard_normal(2)
                u = u / np.linalg.norm(u)
                dpos = OFFSET * u
            vg_new = np.zeros((3, 1))
            vg_new[fate_g, 0] = ORGAN_BIAS
            vg_new = vg_new + rng.standard_normal((3, 1)) / 8.0
            Hp, Hs = Hb["g"]
            Hb["g"] = (np.concatenate([Hp, Hp[:, [im], :] + dpos[None, :, None].transpose(1, 0, 2)], axis=1),
                       np.concatenate([Hs, np.repeat(softmax(vg_new)[:, :, None], T, axis=2)], axis=1))
            Vb["g"] = np.hstack([Vb["g"], vg_new])
            Vdb["g"] = np.hstack([Vdb["g"], np.zeros((3, 1))])
            Ab["g"] = (np.hstack([Pg, Pg[:, [im]] + dpos[:, None]]), np.hstack([Sg, softmax(vg_new)]))
            gfm = np.zeros((3, 1))
            gfm[fate_g, 0] = V_BIAS                                          # her fate, kept as a prior at e^1
            g_pmean = np.hstack([g_pmean, gfm])
            g_pprec = np.append(g_pprec, ORGAN_FATE_PREC)
            knew = int(L.NG)
            rec["organ_divisions"].append((iY, im, knew, int(gtypes[im]), fate_g))
            last_organ_div = iY
            streak_o = 0
            # her zygote: one cell, determined internal, its view of its ensemble her fate
            vl0 = np.zeros((3, 1))
            vl0[0, 0] = V_BIAS
            vc0 = np.zeros((3, 1))
            vc0[fate_g, 0] = ORGAN_BIAS
            newv = {"l": vl0 + rng.standard_normal((3, 1)) / 8.0, "c": vc0 + rng.standard_normal((3, 1)) / 8.0}
            add_cell(Vb, Vdb, Ab, Hb, T, newv, {"c": np.zeros((3, 1)), "s": np.zeros((3, 1))}, None, None)
            born = np.append(born, iY)
            last_div = np.append(last_div, iY)
            prior_mean_l = np.hstack([prior_mean_l, vl0])
            pprec = np.append(pprec, PRIOR_PREC)
            own = np.append(own, knew)
            N = N + 1
            anchor[knew], streak_p[knew], last_parent_div[knew] = N - 1, 0, -10 ** 6
            streak_k[knew], last_prune[knew] = 0, -10 ** 6
            rec["celled_births"].append((iY, knew, int(fate_g)))
            rebuild(Vb, Vdb, Ab, Hb, T)

    n, ny = fx.n, fx.ny
    F = -EiSE / 2.0 + Hquc / 2.0 + fx.logdet_iS * nY / 2.0 - n * ny * nY * np.log(2 * np.pi) / 2.0
    Vb, Ab = L.v_unvec(V), L.a_unvec(a)
    return dict(rec=rec, V=Vb, A=Ab, F=float(F), own=own.tolist(), NG=int(L.NG), N_final=int(N),
                seconds=round(time.time() - t0, 1),
                params=dict(seed=seed, nY=nY, ablate=ablate, organ_ablate=organ_ablate, mutate=mutate, wall=wall))
