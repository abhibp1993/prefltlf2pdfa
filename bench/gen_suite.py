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

    def add_case(n, num_aps, formula_size, density_label, density_p, seed):
        case_id = make_case_id(n, num_aps, formula_size, density_label, seed)
        if case_id in seen_ids:
            return
        seen_ids.add(case_id)

        rng = random.Random(seed)
        formulas = generate_formulas(n, num_aps, formula_size, seed)
        partial_order = generate_partial_order(n, density_p, rng)

        cases.append({
            "case_id": case_id,
            "n": n,
            "num_aps": num_aps,
            "formula_size": formula_size,
            "density": density_label,
            "density_p": density_p,
            "seed": seed,
            "formulas": formulas,
            "partial_order": partial_order,
        })

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
    print(f"  {len(cases)} cases generated.")

    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Written to {args.output}")


if __name__ == "__main__":
    main()
