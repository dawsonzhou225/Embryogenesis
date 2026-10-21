"""check_same.py -- are two run JSONs the same run?  Compares every trajectory the two files
share, value for value (exact equality, no tolerance), and says where they first differ.

    python3 check_same.py results/base_s0.json other/base_s0.json
    python3 check_same.py new.json old.json --prefix     # a shorter run against a longer one: its first steps

Keys only one file has are ignored; self_cons_by_t is aligned at its end."""
import json, sys

TRAJ = ["N_t", "J_t", "organ_N_t", "organ_emit_t", "pos_t", "types_t", "p_t", "sen_t", "own_t", "views_t",
        "organ_pos_t", "organ_types_t", "organ_sec_t", "organ_sen_t", "self_cons_by_t"]
FINAL = ["F", "N_final", "celled", "own_final", "counts_by_ens", "organ_counts", "organ_types_final",
         "organ_pos_final", "divisions", "celled_births", "organ_divisions", "removals"]


def norm(x):
    """JSON round-trips tuples to lists and int dict keys to strings; compare on that footing."""
    if isinstance(x, dict):
        return {str(k): norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    return x


def main(new_path, old_path, prefix=False):
    new, old = json.load(open(new_path)), json.load(open(old_path))
    if new.get("error") or old.get("error"):
        print("a run failed:", new.get("error"), old.get("error"))
        return 1
    bad = 0
    for k in TRAJ:
        if k not in new or k not in old:
            continue
        a, b = norm(new[k]), norm(old[k])
        if k == "self_cons_by_t":                       # the old code started this record late
            a = a[len(old["N_t"]) - len(b):]
        if prefix:
            a, b = a[:len(b)], b[:len(a)]
        if len(a) != len(b):
            print(f"{k}: lengths differ, {len(a)} vs {len(b)}")
            bad += 1
            continue
        first = next((t for t in range(len(a)) if a[t] != b[t]), None)
        if first is None:
            print(f"{k}: same, {len(a)} steps")
        else:
            print(f"{k}: DIFFERS from step {first + 1} of {len(a)}")
            bad += 1
    for k in ([] if prefix else FINAL):
        if k not in new or k not in old:
            continue
        if norm(new[k]) == norm(old[k]):
            print(f"{k}: same")
        else:
            print(f"{k}: DIFFERS  new {str(norm(new[k]))[:80]}  old {str(norm(old[k]))[:80]}")
            bad += 1
    print(("\nSAME RUN" if not prefix else "\nSAME RUN over the shared steps") if bad == 0 else f"\n{bad} differences")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2], prefix="--prefix" in sys.argv[3:]) else 0)
