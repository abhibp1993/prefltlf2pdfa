"""
gen_suite.py — Generate the benchmark manifest (JSON list of cases).

Usage:
    python gen_suite.py --output suites/suite_TIMESTAMP.json

Each case dict:
    {
        "case_id": str,        # unique identifier
        "n": int,              # number of formulas
        "num_aps": int,        # number of atomic propositions
        "formula_size": int,   # max operator count (spot randltl tree_size)
        "density": str,        # "sparse" | "medium" | "dense" | "total"
        "density_p": float,    # edge probability used
        "seed": int,           # random seed
        "formulas": list[str], # LTLf formula strings
        "partial_order": list  # list of [i, j] pairs (i weakly preferred to j)
    }
"""

import argparse
import json
import random
import re

import numpy as np
import spot
from ltlf2dfa.parser.ltlf import LTLfParser

_LTLF_PARSER = LTLfParser()
_MAX_RETRIES = 20   # extra seeds to try before giving up on a parameter combination

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

N_VALUES = [2, 5, 10, 15, 20, 25, 30]
NUM_APS_VALUES = [2, 3, 5, 8, 10]
FORMULA_SIZE_VALUES = [3, 5, 10, 15, 20]
DENSITIES = [
    ("sparse", 0.2),
    ("medium", 0.5),
    ("dense", 0.8),
    ("total", 1.0),
]
SEEDS = [0, 1, 2, 3, 4]

BASELINE = {"n": 5, "num_aps": 3, "formula_size": 5, "density": "medium", "density_p": 0.5}


# ---------------------------------------------------------------------------
# Formula generation
# ---------------------------------------------------------------------------

_AP_NAMES = list("abcdefghijklmnopqrstuvwxyz")


def generate_formulas(n: int, num_aps: int, formula_size: int, seed: int) -> list:
    """Generate n random LTLf formulas with at most formula_size operators."""
    aps = _AP_NAMES[:num_aps]
    formulas = []
    gen = spot.randltl(aps, n, seed=seed, tree_size=formula_size)
    gen = gen.simplify().unabbreviate("WMR")
    _SPOT_BOOL_MAP = {"1": "true", "0": "false"}
    for f in gen:
        s = str(f)
        s = _SPOT_BOOL_MAP.get(s, s)
        # ltlf2dfa requires spaces between unary temporal ops and their operands
        # e.g. spot outputs "Fa" but the parser needs "F a"
        s = re.sub(r'([FGX])([a-z(])', r'\1 \2', s)
        formulas.append(s)
    # Pad with "true" if spot returns fewer than n formulas (rare edge case)
    while len(formulas) < n:
        formulas.append("true")
    return formulas[:n]


# ---------------------------------------------------------------------------
# Partial order generation
# ---------------------------------------------------------------------------

def generate_partial_order(n: int, p: float, rng: random.Random) -> list:
    """
    Generate a random partial order on n elements with edge density p.
    Returns a list of [i, j] pairs meaning formula i is weakly preferred to j (i >= j).
    Diagonal (reflexive) entries are excluded from the output.
    """
    mat = np.zeros((n, n), dtype=bool)

    # Fill upper triangle with probability p
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                mat[i, j] = True

    # Enforce transitivity (Floyd-Warshall)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                mat[i, j] = mat[i, j] or (mat[i, k] and mat[k, j])

    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j]:
                edges.append([i, j])
    return edges


# ---------------------------------------------------------------------------
# Spec string builder
# ---------------------------------------------------------------------------

def build_spec_string(formulas: list, partial_order: list) -> str:
    """Build the prefltlf spec string from formula strings and partial order edges."""
    n = len(formulas)
    lines = [f"prefltlf {n}", ""]
    for f in formulas:
        lines.append(f)
    lines.append("")
    for i, j in partial_order:
        lines.append(f">=, {i}, {j}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consistency validation
# ---------------------------------------------------------------------------

def is_consistent(formulas: list, partial_order: list) -> tuple:
    """
    Return (True, "") if the spec is consistent, else (False, reason_string).

    Consistency checks (no MONA required):
      1. Every formula is syntactically valid LTLf (lark parse).
      2. No two formulas are identical strings (duplicate formulas lead to
         ambiguous preference relations).
      3. The preference relation contains no contradictions:
           - no cycle  (i >= j AND j >= i AND i != j with strict resolution)
           - no pair in both weak-preference and incomparable
         Note: with our upper-triangle generator (only ">=" edges, no "<>"),
         the relation is always structurally consistent; this check catches
         any future generator changes.
    """
    # Check 1: syntactic validity of each formula
    for f in formulas:
        try:
            _LTLF_PARSER(f)
        except Exception as exc:
            return False, f"formula parse error '{f}': {exc}"

    # Check 2: no duplicate formulas
    if len(set(formulas)) < len(formulas):
        seen, dups = set(), []
        for f in formulas:
            if f in seen:
                dups.append(f)
            seen.add(f)
        return False, f"duplicate formulas: {dups}"

    # Check 3: no cycles in the preference relation
    # Build adjacency for the given >= edges.
    # A cycle exists if i >= j AND j >= i (after any transitive chain).
    n = len(formulas)
    # reachable[i] = set of j reachable from i via >= edges
    reachable = {i: set() for i in range(n)}
    for i, j in partial_order:
        reachable[i].add(j)
    # Transitive closure (small n, simple BFS)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for mid in list(reachable[i]):
                new = reachable[mid] - reachable[i]
                if new:
                    reachable[i] |= new
                    changed = True
    for i in range(n):
        if i in reachable[i]:
            return False, f"cyclic preference: formula {i} is reachable from itself"

    return True, ""


# ---------------------------------------------------------------------------
# Case ID
# ---------------------------------------------------------------------------

def make_case_id(n, num_aps, formula_size, density, seed):
    return f"n{n}_ap{num_aps}_sz{formula_size}_{density}_s{seed}"


# ---------------------------------------------------------------------------
# Suite generation
# ---------------------------------------------------------------------------

def generate_suite() -> list:
    cases = []
    seen_ids = set()
    skipped = 0

    def add_case(n, num_aps, formula_size, density_label, density_p, nominal_seed):
        """
        Try to generate a consistent case for (n, num_aps, formula_size, density, nominal_seed).
        If the nominal seed produces an inconsistent spec, try up to _MAX_RETRIES additional
        seeds. The stored seed is always the nominal seed so the case_id is deterministic;
        the actual generation seed used is recorded in the case dict.
        """
        nonlocal skipped
        case_id = make_case_id(n, num_aps, formula_size, density_label, nominal_seed)
        if case_id in seen_ids:
            return
        seen_ids.add(case_id)

        for attempt in range(_MAX_RETRIES + 1):
            actual_seed = nominal_seed + attempt * 1000   # shift seed on retry
            rng = random.Random(actual_seed)
            formulas = generate_formulas(n, num_aps, formula_size, actual_seed)
            partial_order = generate_partial_order(n, density_p, rng)

            ok, reason = is_consistent(formulas, partial_order)
            if ok:
                cases.append({
                    "case_id": case_id,
                    "n": n,
                    "num_aps": num_aps,
                    "formula_size": formula_size,
                    "density": density_label,
                    "density_p": density_p,
                    "seed": nominal_seed,
                    "actual_seed": actual_seed,
                    "formulas": formulas,
                    "partial_order": partial_order,
                })
                if attempt > 0:
                    print(f"  [retry {attempt}] {case_id} — consistent on seed {actual_seed}")
                return

            print(f"  [skip {attempt+1}/{_MAX_RETRIES+1}] {case_id}: {reason}")

        print(f"  [GIVE UP] {case_id}: no consistent spec found after {_MAX_RETRIES+1} attempts")
        skipped += 1

    # Sweep n (hold num_aps, formula_size at baseline; vary density)
    for n in N_VALUES:
        for density_label, density_p in DENSITIES:
            for seed in SEEDS:
                add_case(n, BASELINE["num_aps"], BASELINE["formula_size"],
                         density_label, density_p, seed)

    # Sweep num_aps (hold n, formula_size at baseline; vary density)
    for num_aps in NUM_APS_VALUES:
        for density_label, density_p in DENSITIES:
            for seed in SEEDS:
                add_case(BASELINE["n"], num_aps, BASELINE["formula_size"],
                         density_label, density_p, seed)

    # Sweep formula_size (hold n, num_aps at baseline; vary density)
    for formula_size in FORMULA_SIZE_VALUES:
        for density_label, density_p in DENSITIES:
            for seed in SEEDS:
                add_case(BASELINE["n"], BASELINE["num_aps"], formula_size,
                         density_label, density_p, seed)

    if skipped:
        print(f"  WARNING: {skipped} parameter combinations skipped (no consistent spec found).")
    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark manifest JSON.")
    parser.add_argument("--output", required=True, help="Path to output JSON file.")
    args = parser.parse_args()

    print("Generating benchmark suite...")
    cases = generate_suite()
    retried = sum(1 for c in cases if c.get("actual_seed") != c["seed"])
    print(f"  {len(cases)} cases generated ({retried} needed seed retry).")

    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Written to {args.output}")


if __name__ == "__main__":
    main()
