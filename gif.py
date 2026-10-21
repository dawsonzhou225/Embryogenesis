#!/usr/bin/env python3
"""gif.py -- GIFs from traced runs: blanket 0 in its own frame, or the whole organ.

    python3 gif.py results/abl_core_s0.json --from 144 --out abl_core.gif --title "blanket 0: its core removed at 150"
    python3 gif.py results/abl_blanket2_s0.json --organ --from 144 --out abl_blanket2.gif
    python3 gif.py results/abl_core_s0.json results/dev_core_s0.json --from 1 --out cores.gif    # one panel per run
    python3 gif.py --suite results                                   # the five GIFs of figures_and_animations/ into results/gifs/

Cells are coloured by kind (red core, blue active, grey sensory); dotted circles at radius 1 and 2
from the core; + the blanket's centroid; an orange line is a signalling wall while it stands.  The
organ view draws cells at their ensemble's position + ZOOM x their local position.
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COL = {0: "#d62728", 1: "#1f77b4", 2: "#7f7f7f"}
ZOOM = 0.15


def blanket0(d, t):
    own, ty = np.array(d["own_t"][t]), np.array(d["types_t"][t])
    P, p = np.array(d["pos_t"][t]), np.array(d["p_t"][t])
    idx = np.flatnonzero(own == 0)
    has_core = (ty[idx] == 0).any()
    core = P[:, idx[np.argmax(p[0, idx])]] if has_core else P[:, idx].mean(axis=1)
    return P[:, idx], ty[idx], core, has_core


def wall_x(d, t):
    """The wall's x in blanket 0's frame at step index t, or None (blanket 0 is the organ's core)."""
    ev = d.get("wall") or []
    up = [e for e in ev if e[2] == "up"]
    if not up:
        return None
    down = [e for e in ev if e[2] == "down"]
    if up[0][0] <= t + 1 < (down[0][0] if down else 10 ** 9):
        return (float(up[0][1]) - float(d["organ_pos_t"][t][0][0])) / 0.05
    return None


def draw_blanket(ax, d, t, centre, title, lim=2.8):
    P, ty, c, has_core = blanket0(d, t)
    ax.cla()
    for k in (2, 1, 0):
        m = ty == k
        ax.plot(P[0, m], P[1, m], "o", color=COL[k], ms=7 if k == 0 else 5)
    if has_core:
        for r in (1, 2):
            ax.add_patch(plt.Circle(c, r, fill=False, ls=":", lw=0.7, color="k"))
    ax.plot(*P.mean(axis=1), "k+", ms=8, mew=1)
    x = wall_x(d, t)
    if x is not None:
        ax.axvline(x, color="#ff7f0e", lw=2, alpha=0.7)
    cnt = [int((ty == k).sum()) for k in range(3)]
    ax.set_aspect("equal"); ax.set_xlim(centre[0] - lim, centre[0] + lim); ax.set_ylim(centre[1] - lim, centre[1] + lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title((f"{title}\n" if title else "") + f"step {t + 1}: ({cnt[0]},{cnt[1]},{cnt[2]})", fontsize=10)


def draw_organ(ax, d, t, centre, title, lim=2.8):
    Pg, gt = np.array(d["organ_pos_t"][t]), np.array(d["organ_types_t"][t])
    own, ty, P = np.array(d["own_t"][t]), np.array(d["types_t"][t]), np.array(d["pos_t"][t])
    ax.cla()
    for k in range(len(gt)):
        ax.add_patch(plt.Circle(Pg[:, k], 0.33, color=COL[int(gt[k])], alpha=0.12, lw=0))
    X = Pg[:, own] + ZOOM * P
    for k in (2, 1, 0):
        m = ty == k
        ax.plot(X[0, m], X[1, m], "o", color=COL[k], ms=2.5 if k else 3.5)
    for r in (1, 2):
        ax.add_patch(plt.Circle(Pg[:, 0], r, fill=False, ls=":", lw=0.7, color="k"))
    ax.set_aspect("equal"); ax.set_xlim(centre[0] - lim, centre[0] + lim); ax.set_ylim(centre[1] - lim, centre[1] + lim)
    ax.set_xticks([]); ax.set_yticks([])
    oc = [int((gt == k).sum()) for k in range(3)]
    ax.set_title((f"{title}\n" if title else "") + f"step {t + 1}: {len(gt)} ensembles ({oc[0]},{oc[1]},{oc[2]}), {own.size} cells", fontsize=10)


def make(runs, out, t0=1, t1=None, every=1, fps=12, suptitle="", hold=6):
    """runs: list of (json path, 'blanket' | 'organ', panel title)."""
    data = [(json.load(open(p)), kind, ttl) for p, kind, ttl in runs]
    T = min(len(d["types_t"]) for d, _, _ in data)
    t1 = min(t1 or T, T)
    steps = list(range(t0 - 1, t1, every)) + [t1 - 1] * hold
    width = max(4.2 * len(runs), 6.4)
    fig, axes = plt.subplots(1, len(runs), figsize=(width, 4.6))
    fig.subplots_adjust(top=0.80)
    axes = np.atleast_1d(axes)
    centres = [np.median([blanket0(d, t)[0].mean(axis=1) for t in steps], axis=0) if kind == "blanket"
               else np.array(d["organ_pos_t"][steps[0]])[:, 0] for d, kind, _ in data]     # a still window

    def draw(t):
        for ax, (d, kind, ttl), c in zip(axes, data, centres):
            (draw_blanket if kind == "blanket" else draw_organ)(ax, d, t, c, ttl)
        fig.suptitle(textwrap.fill(suptitle, int(14 * width)), fontsize=10)
    FuncAnimation(fig, draw, frames=steps, interval=1000 / fps).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print("wrote", out, f"({len(steps)} frames)")


def suite(res, seed, every, fps):
    g = os.path.join(res, "gifs")
    os.makedirs(g, exist_ok=True)
    j = lambda n: os.path.join(res, f"{n}_s{seed}.json")
    make([(j("abl_blanket2"), "organ", "organ: 2 ensembles removed"), (j("abl_core"), "blanket", "blanket 0: its core removed"),
          (j("abl_sen4"), "blanket", "blanket 0: 4 sensory cells removed")],
         os.path.join(g, "regeneration_three_levels.gif"), 144, 256, every, fps, "Regeneration at three levels from one deficit rule (event at step 150)")
    make([(j("abl_core"), "blanket", "core removed at 150 (formed)"), (j("dev_core"), "blanket", "core removed at 60 (growing)")],
         os.path.join(g, "regeneration_young_vs_formed.gif"), 1, 256, every, fps, "The same loss during growth and after it")
    make([(j("mut_hold"), "blanket", "")], os.path.join(g, "mutation_repaired.gif"), 144, 256, every, fps,
         "Blanket 0: 3 cells held on the wrong kind at step 150. A mutation the cell cannot shed: the misfits are replaced one by one")
    make([(j("wall_late"), "blanket", "")], os.path.join(g, "wall_healed.gif"), 50, 256, every, fps,
         "Blanket 0: a signalling wall (orange) from step 60 to 140. Inner cells cut off from half the field drift out and relabel; each is swapped out and the ring recovers")
    make([(j("mut_flip"), "blanket", "3 cells' belief flipped at 150")], os.path.join(g, "flip_corrected.gif"), 146, 160, 1, 3,
         "A flipped identity is overwritten by the field within the step", hold=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="traced JSONs, one panel each")
    ap.add_argument("--organ", action="store_true", help="draw the whole organ instead of blanket 0")
    ap.add_argument("--from", dest="t0", type=int, default=1); ap.add_argument("--to", dest="t1", type=int)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--title", default=""); ap.add_argument("--out", default="run.gif")
    ap.add_argument("--suite", metavar="RESULTS", help="make the five GIFs of figures_and_animations/ from this results folder")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.suite:
        suite(a.suite, a.seed, a.every, a.fps)
    elif a.runs:
        make([(p, "organ" if a.organ else "blanket", os.path.basename(p).replace(".json", "")) for p in a.runs],
             a.out, a.t0, a.t1, a.every, a.fps, a.title)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
