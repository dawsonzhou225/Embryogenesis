"""run.py -- run the model over seeds, one JSON per run (every step traced), and report.

    python3 run.py run --name base --seeds 0-4 --cores 3 --out results
    python3 run.py run --name abl_core --ablate-at 150 --ablate-what core ...
    python3 run.py run --name wall_dev --wall-at 20 --wall-until 100 ...
    python3 run.py report --out results

Perturbations (all on blanket 0, the organ's core): --ablate-at/--ablate-what/--ablate-n,
--organ-ablate-at/--organ-ablate-n, --mutate-at/--mutate-n/--mutate-what, --wall-at/--wall-until.
Runs already in --out are skipped.  Each run is a subprocess with one BLAS thread.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PERTURB = ["ablate_at", "ablate_what", "ablate_n", "organ_ablate_at", "organ_ablate_n",
           "mutate_at", "mutate_n", "mutate_what", "wall_at", "wall_until"]


def job_path(out, name, seed):
    return os.path.join(out, f"{name}_s{seed}.json")


def one(a):
    """One run -> JSON: a summary, the final state, and the per-step traces."""
    sys.path.insert(0, HERE)
    import numpy as np
    import model
    kw = dict(seed=a.seed, nY=a.nY,
              ablate=(a.ablate_at, a.ablate_what, a.ablate_n) if a.ablate_at else None,
              organ_ablate=(a.organ_ablate_at, a.organ_ablate_n) if a.organ_ablate_at else None,
              mutate=(a.mutate_at, a.mutate_n, a.mutate_what) if a.mutate_at else None,
              wall=(a.wall_at, a.wall_until) if a.wall_at else None)
    t0 = time.time()
    err, o = None, None
    try:
        o = model.run(**kw)
    except FloatingPointError as e:
        err = str(e)
    params = dict(nY=a.nY, **{k: getattr(a, k) for k in PERTURB})
    res = dict(name=a.name, seed=a.seed, params=params, error=err, seconds=round(time.time() - t0, 1))
    if o is not None:
        r, own = o["rec"], np.array(o["own"])
        ty = model.softmax(o["V"]["l"]).argmax(axis=0)
        views = model.softmax(o["V"]["c"]).argmax(axis=0)
        gty = model.softmax(o["V"]["g"]).argmax(axis=0)
        counts_by = {int(k): [int(((ty == t) & (own == k)).sum()) for t in range(3)] for k in range(o["NG"])}
        res.update(
            N_final=o["N_final"], F=o["F"], K_final=o["NG"], own_final=own.tolist(), counts_by_ens=counts_by,
            blankets_ok=sum(c == [1, 6, 9] for c in counts_by.values()),
            organ_counts=[int((gty == t).sum()) for t in range(3)], views_agree=float((views == gty[own]).mean()),
            self_cons_mean=float(np.mean(list(r["self_cons_by"][-1].values()))),
            divisions=r["divisions"], celled_births=r["celled_births"], organ_divisions=r["organ_divisions"],
            removals=r["removals"], organ_removals=r["organ_removals"], mutations=r["mutations"], wall=r["wall"],
            reanchor_steps=r["reanchor"],
            N_t=r["N"], J_t=r["J"], organ_N_t=r["organ_N"], organ_emit_t=r["organ_emit"],
            pos_t=r["pos"], types_t=r["types"], p_t=r["p"], sen_t=r["sen"], own_t=r["own"], views_t=r["views"],
            organ_pos_t=r["organ_pos"], organ_types_t=r["organ_types"], organ_sec_t=r["organ_sec"], organ_sen_t=r["organ_sen"],
            self_cons_by_t=r["self_cons_by"], deficit_by_t=r["deficit_by"], err_by_t=r["err_by"],
            organ_types_final=gty.tolist(), organ_pos_final=np.round(o["A"]["g"][0], 3).tolist(), types_final=ty.tolist())
    os.makedirs(a.out, exist_ok=True)
    with open(job_path(a.out, a.name, a.seed), "w") as f:
        json.dump(res, f)


def launch(a, seed):
    env = dict(os.environ, **{v: "1" for v in THREAD_VARS})
    cmd = [sys.executable, os.path.abspath(__file__), "one", "--name", a.name, "--seed", str(seed), "--out", a.out, "--nY", str(a.nY)]
    for k in PERTURB:
        if getattr(a, k) is not None:
            cmd += ["--" + k.replace("_", "-"), str(getattr(a, k))]
    t = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return seed, r.returncode, time.time() - t, r.stderr[-1500:]


def run_sweep(a):
    os.makedirs(a.out, exist_ok=True)
    seeds = [s for s in a.seeds if not os.path.exists(job_path(a.out, a.name, s))]
    if not seeds:
        print(f"{a.name}: every requested result already exists")
        return
    workers = min(a.cores or os.cpu_count() or 1, len(seeds))
    print(f"{a.name}: {len(seeds)} runs, {workers} at once, nY = {a.nY}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(launch, a, s) for s in seeds]
        for f in as_completed(futs):
            s, rc, dt, err = f.result()
            if rc != 0:
                print(f"   seed {s:<3} FAILED (exit {rc})\n{err}", flush=True)
                continue
            r = json.load(open(job_path(a.out, a.name, s)))
            if r.get("error"):
                print(f"   seed {s:<3} {dt:5.0f}s   FAILED in the solver: {r['error']}", flush=True)
            else:
                oc = r["organ_counts"]
                print(f"   seed {s:<3} {dt:5.0f}s   organ ({oc[0]},{oc[1]},{oc[2]})  blankets (1,6,9): {r['blankets_ok']}/{r['K_final']}  "
                      f"blanket 0 {tuple(r['counts_by_ens']['0'])}  views = kind {r['views_agree']:.2f}", flush=True)


def report(a):
    import numpy as np
    rows = [json.load(open(os.path.join(a.out, f))) for f in sorted(os.listdir(a.out)) if f.endswith(".json")]
    groups = {}
    for r in rows:
        groups.setdefault(r["name"], []).append(r)
    for name, rs in groups.items():
        ok = [r for r in rs if not r.get("error")]
        tally = {}
        for r in ok:
            c = tuple(r["counts_by_ens"]["0"])
            tally[c] = tally.get(c, 0) + 1
        b0 = "  ".join(f"({c[0]},{c[1]},{c[2]})x{n}" for c, n in sorted(tally.items(), key=lambda kv: -kv[1]))
        print(f"{name}: {len(ok)} runs ({len(rs) - len(ok)} failed), {ok[0]['params']['nY']} steps, {np.mean([r['seconds'] for r in ok]):.0f} s each")
        print(f"  organ (1,6,9): {sum(r['organ_counts'] == [1, 6, 9] for r in ok)}/{len(ok)}   "
              f"blankets (1,6,9): {sum(r['blankets_ok'] for r in ok)}/{sum(r['K_final'] for r in ok)}   "
              f"blanket 0: {b0}   views = kind: {np.mean([r['views_agree'] for r in ok]):.2f}   "
              f"turnovers/run: {np.mean([sum(x[3] == 'turnover' for x in r['removals']) for r in ok]):.1f}")


def parse_seeds(tokens):
    out = []
    for t in tokens:
        lo, hi = t.split("-") if "-" in t else (t, t)
        out += list(range(int(lo), int(hi) + 1))
    return out


def add_run_args(p):
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--nY", type=int, default=256)
    p.add_argument("--ablate-at", type=int); p.add_argument("--ablate-what", default="sensory", choices=["core", "sensory", "active", "random"])
    p.add_argument("--ablate-n", type=int, default=4)
    p.add_argument("--organ-ablate-at", type=int); p.add_argument("--organ-ablate-n", type=int, default=2)
    p.add_argument("--mutate-at", type=int); p.add_argument("--mutate-n", type=int, default=3)
    p.add_argument("--mutate-what", default="hold", choices=["flip", "hold"])
    p.add_argument("--wall-at", type=int); p.add_argument("--wall-until", type=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); add_run_args(r)
    r.add_argument("--seeds", nargs="+", default=["0-4"]); r.add_argument("--cores", type=int, default=None)
    r.set_defaults(fn=run_sweep)
    o = sub.add_parser("one"); add_run_args(o)
    o.add_argument("--seed", type=int, required=True); o.set_defaults(fn=one)
    p = sub.add_parser("report"); p.add_argument("--out", default="results"); p.set_defaults(fn=report)
    a = ap.parse_args()
    if hasattr(a, "seeds"):
        a.seeds = parse_seeds(a.seeds)
    a.fn(a)


if __name__ == "__main__":
    main()
