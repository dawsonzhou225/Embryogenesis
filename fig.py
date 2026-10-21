#!/usr/bin/env python3
"""fig.py -- the suite figure: every config in a results folder, mean over seeds with the min-max band.

Panels: sortedness of the blankets (active-sensory pairs with the active nearer the core); blanket 0's
radial error (mean |r - r_kind| over its ring cells, r_kind = 1 active, 2 sensory, from its core) and
the unevenness of its sensory ring (largest gap between neighbours over the ideal 2 pi / 9); views =
kind; the consensus gap; the blanket and organ counts' distance from (1,6,9); the residual prediction
error per cell by block; cells; ensembles.  Also a recovery table: steps after the event until each
curve is back inside the control's band for the rest of the run.

    python3 fig.py --out results                        # -> results/suite.png, results/recovery.txt
    python3 fig.py --out results --configs base abl_core --control base --name regen
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RING_R = np.array([0.0, 1.0, 2.0])
TARGET = np.array([1, 6, 9])
PANELS = [("sorted", "sortedness of the blankets", (0.9, 1.005)), ("radial0", "blanket 0: radial error |r - r_kind|", (0, 0.4)),
          ("uneven0", "blanket 0: largest sensory gap / ideal gap", (0.9, 3)), ("views", "views = kind", (0.8, 1.005)),
          ("cons", "consensus gap", (0, 0.05)), ("d169_blankets", "blanket counts: |counts - (1,6,9)|", (0, 3)),
          ("d169_organ", "organ counts: |counts - (1,6,9)|", (0, 4)), ("err_lsen", "residual error per cell: local field (l_sen)", (0, 0.1)),
          ("err_ssec", "residual error per cell: self-model accuracy (s_sec)", (0, 0.004)),
          ("err_csen", "residual error per cell: organ field (c_sen)", (0, 0.1)), ("N", "cells", None), ("NG", "ensembles", None)]


def load(out):
    groups = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(out, f)))
            if not r.get("error") and r.get("pos_t"):
                groups.setdefault(r["name"], []).append(r)
    return groups


def per_step(r):
    T = len(r["pos_t"])
    out = {k: [] for k, _, _ in PANELS}
    out["N"], out["NG"] = r["N_t"], r["organ_N_t"]
    for t in range(T):
        P, ty, own = np.array(r["pos_t"][t]), np.array(r["types_t"][t]), np.array(r["own_t"][t])
        views, gty, p = np.array(r["views_t"][t]), np.array(r["organ_types_t"][t]), np.array(r["p_t"][t])
        sc, d169 = [], []
        for k in range(len(gty)):
            idx = np.flatnonzero(own == k)
            core = idx[np.argmax(p[0, idx])]
            rr = np.linalg.norm(P[:, idx] - P[:, [core]], axis=0)
            ra, rs = rr[ty[idx] == 1], rr[ty[idx] == 2]
            sc.append(float((ra[:, None] < rs[None, :]).mean()) if ra.size and rs.size else 1.0)
            d169.append(float(np.abs(np.array([(ty[idx] == j).sum() for j in range(3)]) - TARGET).sum()))
            if k == 0:
                rel, kk = P[:, idx] - P[:, [core]], ty[idx]
                ring = kk > 0
                out["radial0"].append(float(np.abs(rr[ring] - RING_R[kk[ring]]).mean()) if ring.any() else 0.0)
                sens = kk == 2
                if sens.sum() >= 3:
                    ang = np.sort(np.arctan2(rel[1, sens], rel[0, sens]))
                    gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
                    out["uneven0"].append(float(gaps.max() / (2 * np.pi / sens.sum())))
                else:
                    out["uneven0"].append(np.nan)
        out["sorted"].append(float(np.mean(sc)))
        out["d169_blankets"].append(float(np.mean(d169)))
        out["d169_organ"].append(float(np.abs(np.array([(gty == j).sum() for j in range(3)]) - TARGET).sum()))
        out["views"].append(float((views == gty[own]).mean()))
        out["cons"].append(float(np.mean(list(r["self_cons_by_t"][t].values()))))
        e, nN = r["err_by_t"][t], max(len(ty), 1)
        out["err_lsen"].append(e["l_sen"] / nN); out["err_ssec"].append(e["s_sec"] / nN); out["err_csen"].append(e["c_sen"] / nN)
    return {k: np.asarray(v, float) for k, v in out.items()}


def event_step(r):
    p = r["params"]
    return next((int(p[k]) for k in ("mutate_at", "ablate_at", "organ_ablate_at", "wall_at") if p.get(k)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--control", default="base")
    ap.add_argument("--name", default="suite")
    a = ap.parse_args()
    groups = load(a.out)
    names = a.configs or list(groups)
    if a.control in groups and a.control not in names:
        names = [a.control] + list(names)
    curves = {n: [per_step(r) for r in groups[n]] for n in names if groups.get(n)}
    events = {n: event_step(groups[n][0]) for n in curves}
    walls = {n: groups[n][0].get("wall", []) for n in curves}
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    for ax, (key, title, ylim) in zip(axes.ravel(), PANELS):
        for ci, n in enumerate(curves):
            T = min(len(c[key]) for c in curves[n])
            M = np.array([c[key][:T] for c in curves[n]])
            x = np.arange(1, T + 1)
            ax.plot(x, np.nanmean(M, axis=0), color=cols[ci % 10], label=f"{n} (n={len(M)})", lw=1.4)
            ax.fill_between(x, np.nanmin(M, axis=0), np.nanmax(M, axis=0), color=cols[ci % 10], alpha=0.15, lw=0)
            if events[n]:
                ax.axvline(events[n], color=cols[ci % 10], ls=":", lw=0.9)
            for w in walls[n]:
                ax.axvline(w[0], color=cols[ci % 10], ls="--", lw=0.8)
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("step")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{a.out}: dotted the event, dashed a wall up / down, band min to max over seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.name}.png"), dpi=110)
    print("wrote", os.path.join(a.out, f"{a.name}.png"))

    ctrl = curves.get(a.control)
    lines = []
    for n in curves:
        ev = events[n]
        if not ev or ctrl is None or n == a.control:
            continue
        row = [f"{n:14s} event {ev:4d}"]
        for key in ("sorted", "radial0", "uneven0", "d169_blankets", "d169_organ", "views", "cons", "NG"):
            T = min(min(len(c[key]) for c in curves[n]), min(len(c[key]) for c in ctrl))
            lo = np.nanmin(np.array([c[key][:T] for c in ctrl]), axis=0) - 1e-9
            hi = np.nanmax(np.array([c[key][:T] for c in ctrl]), axis=0) + 1e-9
            back = []
            for c in curves[n]:
                y = c[key][:T]
                inside = (y >= lo) & (y <= hi) | np.isnan(y)
                back.append(next((t + 1 - ev for t in range(ev - 1, T) if inside[t:].all()), None))
            got = [b for b in back if b is not None]
            row.append(f"{key}: {np.mean(got):.0f} ({len(got)}/{len(back)})" if got else f"{key}: never (0/{len(back)})")
        lines.append("  ".join(row))
    txt = "steps after the event until the curve is back inside the control's min-max band for the rest of the run\n" + "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, f"{a.name}_recovery.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
