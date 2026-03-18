# Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `bench/` scalability benchmark suite for `PrefLTLf.translate()` as specified in `docs/plans/2026-03-18-benchmark-design.md`.

**Architecture:** Subprocess-isolated harness: `gen_suite.py` pre-generates a JSON manifest of all cases; `run_bench.py` spawns one `worker.py` subprocess per case with timeout + WSL RAM cap; `worker.py` instruments per-stage timing via `time.perf_counter()` and peak memory via `tracemalloc`; `analyze.ipynb` reads the CSV and produces paper-quality plots.

**Tech Stack:** Python 3.10+, `spot` (formula generation), `numpy` (partial order generation), `resource` stdlib (RAM cap, WSL/Linux only), `tracemalloc` stdlib (memory), `pandas` + `matplotlib` + `seaborn` (analysis).

---

## Task 1: Directory scaffold + gitignore

**Files:**
- Create: `bench/suites/.gitkeep`
- Create: `bench/results/.gitkeep`
- Modify: `.gitignore`

**Step 1: Create directories**

```bash
mkdir -p bench/suites bench/results
touch bench/suites/.gitkeep bench/results/.gitkeep
```

**Step 2: Add gitignore entries**

Append to `.gitignore`:
```
bench/suites/*.json
bench/results/*.csv
```

**Step 3: Verify**

```bash
ls bench/suites bench/results
```
Expected: each directory contains `.gitkeep`.

**Step 4: Commit**

```bash
git add bench/suites/.gitkeep bench/results/.gitkeep .gitignore
git commit -m "#add: bench directory scaffold and gitignore entries"
```

---

## Task 2: `bench/gen_suite.py`

**Files:**
- Create: `bench/gen_suite.py`

**Background:** This script generates the benchmark manifest. It uses `spot.randltl` to produce formulas and a density-parameterized partial order generator. The manifest is a JSON list; each entry is one case dict with pre-generated formulas and partial order edges.

The partial order generator (`generate_random_partial_order`) uses `numpy` + transitivity closure. The density parameter `p` controls the probability of adding an edge between any ordered pair `(i, j)` where `i < j`. After filling with probability `p`, transitivity is enforced with Floyd-Warshall.

The spec string format expected by `PrefLTLf` is:
```
prefltlf N

formula_0_str
formula_1_str
...

>=, 0, 1
>=, 0, 2
```

**Step 1: Write `bench/gen_suite.py`**

```python
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
import time

import numpy as np
import spot

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

N_VALUES = [2, 3, 4, 5, 6, 8, 10]
NUM_APS_VALUES = [2, 3, 4, 5]
FORMULA_SIZE_VALUES = [3, 5, 8, 12]
DENSITIES = [
    ("sparse", 0.2),
    ("medium", 0.5),
    ("dense", 0.8),
    ("total", 1.0),
]
SEEDS = [0, 1, 2, 3, 4]

BASELINE = {"n": 4, "num_aps": 3, "formula_size": 5, "density": "medium", "density_p": 0.5}


# ---------------------------------------------------------------------------
# Formula generation
# ---------------------------------------------------------------------------

def generate_formulas(n: int, num_aps: int, formula_size: int, seed: int) -> list[str]:
    """Generate n random LTLf formulas with at most formula_size operators."""
    formulas = []
    gen = spot.randltl(num_aps, n, seed=seed, tree_size=formula_size)
    gen = gen.simplify().unabbreviate("WMR")
    for f in gen:
        formulas.append(str(f))
    # Pad with "true" if spot returns fewer than n formulas (rare edge case)
    while len(formulas) < n:
        formulas.append("true")
    return formulas[:n]


# ---------------------------------------------------------------------------
# Partial order generation
# ---------------------------------------------------------------------------

def generate_partial_order(n: int, p: float, rng: random.Random) -> list[list[int]]:
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

def generate_suite() -> list[dict]:
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
```

**Step 2: Run it with a small smoke-test output**

```bash
cd bench
python gen_suite.py --output suites/smoke.json
```
Expected output: `N cases generated` (approximately 300), file written.

**Step 3: Verify schema of first case**

```bash
python -c "
import json
cases = json.load(open('bench/suites/smoke.json'))
print(f'Total cases: {len(cases)}')
print('First case:')
import pprint; pprint.pprint(cases[0])
"
```
Expected: first case has all required keys, `formulas` is a list of strings, `partial_order` is a list of `[int, int]` pairs.

**Step 4: Commit**

```bash
git add bench/gen_suite.py bench/suites/smoke.json
git commit -m "#add: bench/gen_suite.py - benchmark manifest generator"
```

> Note: `smoke.json` is added here for quick reproducibility checks during development. The `.gitignore` entry only blocks `bench/suites/*.json` — you may want to keep `smoke.json` tracked or adjust the pattern.

---

## Task 3: `bench/worker.py`

**Files:**
- Create: `bench/worker.py`

**Background:** `worker.py` is the subprocess entry point. It:
1. Applies `resource.setrlimit` to cap its own virtual memory (WSL/Linux only)
2. Loads the case from the manifest
3. Builds a `prefltlf` spec string from `formulas` + `partial_order`
4. Instruments each pipeline stage with `time.perf_counter()`
5. Wraps the full `translate()` with `tracemalloc` for peak memory
6. Prints a single JSON line to stdout
7. Exits 0 on success, non-zero on error

**How to build the spec string:**
```
prefltlf N

formula_0
formula_1
...

>=, i, j
>=, i, k
...
```

**How to get per-stage timing without modifying `prefltlf.py`:** Replicate the three steps of `PrefLTLf.translate()` directly in the worker, wrapping each step with `time.perf_counter()`. The internal methods are `_construct_semi_automaton` and `_construct_preference_graph`. The DFA step calls `utils.ltlf2dfa` directly.

**Step 1: Write `bench/worker.py`**

```python
"""
worker.py — Subprocess entry point for one benchmark case.

Invoked by run_bench.py as:
    python worker.py --manifest PATH --case-id ID --mem-limit-mb N

Prints a single JSON line to stdout:
    {"status": "ok", "t_dfa": ..., "t_semi": ..., "t_pref": ...,
     "t_total": ..., "peak_mem_mb": ..., "semi_states": ...,
     "semi_transitions": ..., "pref_nodes": ..., "pref_edges": ...}

On error prints: {"status": "error", "error": "..."}
On OOM the process is killed by the OS before printing anything;
run_bench.py detects this via non-zero exit code + signal.
"""

import argparse
import json
import sys
import time
import tracemalloc

# Apply RAM cap immediately, before importing heavy libraries.
# This must happen before importing prefltlf2pdfa (which imports networkx, etc.)
def _apply_mem_limit(limit_mb: int):
    try:
        import resource
        limit_bytes = limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ImportError, ValueError) as e:
        # resource module unavailable (native Windows) or limit already lower
        print(json.dumps({"status": "error", "error": f"setrlimit failed: {e}"}))
        sys.exit(1)


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


def run_case(case: dict) -> dict:
    """Run translation for one case, returning metrics dict."""
    from prefltlf2pdfa.prefltlf import PrefLTLf, PrefAutomaton
    from prefltlf2pdfa.semantics import semantics_mp_forall_exists
    import prefltlf2pdfa.utils as utils

    spec_str = build_spec_string(case["formulas"], case["partial_order"])
    pref = PrefLTLf(spec_str)

    # Start tracemalloc for peak memory over the full translation
    tracemalloc.start()

    # ---- Stage 1: LTLf → DFA (via MONA) ----
    aut = PrefAutomaton()
    aut.atoms = pref.atoms
    aut.alphabet = pref.alphabet
    aut.phi = pref.phi
    aut.sorted_phi = sorted(pref.phi.keys())

    t0 = time.perf_counter()
    aut.dfa = [utils.ltlf2dfa(pref.phi[i]) for i in aut.sorted_phi]
    t1 = time.perf_counter()

    # ---- Stage 2: Semi-automaton (product construction) ----
    pref._construct_semi_automaton(
        aut=aut,
        show_progress=False,
        use_multiprocessing=False,  # keep single-process inside subprocess
        backend="auto",
        enumeration="auto",
    )
    t2 = time.perf_counter()

    # ---- Stage 3: Preference graph ----
    pref._construct_preference_graph(aut, semantics_mp_forall_exists, show_progress=False)
    t3 = time.perf_counter()

    # Peak memory
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Output size
    semi_states = len(aut.states)
    semi_transitions = sum(len(v) for v in aut.transitions.values())
    pref_nodes = aut.pref_graph.number_of_nodes()
    pref_edges = aut.pref_graph.number_of_edges()

    return {
        "status": "ok",
        "t_dfa": round(t1 - t0, 6),
        "t_semi": round(t2 - t1, 6),
        "t_pref": round(t3 - t2, 6),
        "t_total": round(t3 - t0, 6),
        "peak_mem_mb": round(peak_bytes / (1024 * 1024), 4),
        "semi_states": semi_states,
        "semi_transitions": semi_transitions,
        "pref_nodes": pref_nodes,
        "pref_edges": pref_edges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mem-limit-mb", type=int, default=0,
                        help="RAM cap in MB (0 = no cap).")
    args = parser.parse_args()

    if args.mem_limit_mb > 0:
        _apply_mem_limit(args.mem_limit_mb)

    with open(args.manifest) as f:
        cases = json.load(f)

    case = next((c for c in cases if c["case_id"] == args.case_id), None)
    if case is None:
        print(json.dumps({"status": "error", "error": f"case_id not found: {args.case_id}"}))
        sys.exit(1)

    try:
        result = run_case(case)
        print(json.dumps(result))
        sys.exit(0)
    except MemoryError as e:
        print(json.dumps({"status": "oom", "error": str(e)}))
        sys.exit(1)
    except Exception as e:
        import traceback
        print(json.dumps({"status": "error", "error": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test worker with first case from manifest**

```bash
# Get first case_id from manifest
CASE_ID=$(python -c "import json; print(json.load(open('bench/suites/smoke.json'))[0]['case_id'])")
echo "Testing case: $CASE_ID"

python bench/worker.py \
    --manifest bench/suites/smoke.json \
    --case-id "$CASE_ID" \
    --mem-limit-mb 2048
```
Expected: a single JSON line with `"status": "ok"` and numeric fields.

**Step 3: Verify output parses**

```bash
python bench/worker.py \
    --manifest bench/suites/smoke.json \
    --case-id "$CASE_ID" \
    --mem-limit-mb 2048 | python -m json.tool
```
Expected: pretty-printed JSON with no errors.

**Step 4: Commit**

```bash
git add bench/worker.py
git commit -m "#add: bench/worker.py - per-case translation worker with timing and memory instrumentation"
```

---

## Task 4: `bench/run_bench.py`

**Files:**
- Create: `bench/run_bench.py`

**Background:** The runner iterates the manifest, spawns one `worker.py` subprocess per case, enforces a wall-clock timeout, parses the worker's stdout JSON, and appends one result row to the output CSV. It skips cases whose `case_id` is already in the CSV (resumability). Exit codes from the worker are interpreted as:
- 0 + JSON with `"status": "ok"` → success
- 0 + JSON with `"status": "error"` → recorded as `error`
- Timeout → `subprocess.TimeoutExpired` caught → recorded as `timeout`
- Non-zero exit (OOM signal) → exit code analysis → recorded as `oom` if the worker printed `oom`, else `error`

**Step 1: Write `bench/run_bench.py`**

```python
"""
run_bench.py — Run benchmark suite, one subprocess per case.

Usage:
    python run_bench.py \
        --suite suites/suite.json \
        --output results/results.csv \
        --timeout 300 \
        --mem-limit-mb 4096

Supports resuming: skips case_ids already present in --output CSV.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CSV_FIELDS = [
    "case_id", "n", "num_aps", "formula_size", "density", "seed",
    "status", "t_dfa", "t_semi", "t_pref", "t_total",
    "peak_mem_mb", "semi_states", "semi_transitions", "pref_nodes", "pref_edges",
]


def load_completed(output_path: str) -> set:
    """Return set of case_ids already in the output CSV."""
    if not os.path.exists(output_path):
        return set()
    completed = set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["case_id"])
    return completed


def write_header(output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_row(output_path: str, row: dict):
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def run_case(case: dict, manifest_path: str, timeout: int, mem_limit_mb: int) -> dict:
    """Spawn worker.py for one case; return result dict."""
    worker = Path(__file__).parent / "worker.py"
    cmd = [
        sys.executable, str(worker),
        "--manifest", manifest_path,
        "--case-id", case["case_id"],
        "--mem-limit-mb", str(mem_limit_mb),
    ]

    base = {
        "case_id": case["case_id"],
        "n": case["n"],
        "num_aps": case["num_aps"],
        "formula_size": case["formula_size"],
        "density": case["density"],
        "seed": case["seed"],
    }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout"}

    stdout = result.stdout.strip()

    # Try to parse worker JSON output
    if stdout:
        try:
            metrics = json.loads(stdout)
            if metrics.get("status") == "ok":
                return {**base, **metrics}
            else:
                status = metrics.get("status", "error")
                return {**base, "status": status}
        except json.JSONDecodeError:
            pass

    # Worker printed nothing or garbage — check exit code
    if result.returncode != 0:
        # Distinguish OOM (killed by SIGKILL due to setrlimit) from generic error
        # On Linux, exit code is -9 (SIGKILL) or -11 (SIGSEGV) for OOM via RLIMIT_AS
        if result.returncode in (-9, -11):
            return {**base, "status": "oom"}
        return {**base, "status": "error"}

    return {**base, "status": "error"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="Path to manifest JSON.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case timeout in seconds.")
    parser.add_argument("--mem-limit-mb", type=int, default=4096, help="RAM cap per worker in MB.")
    args = parser.parse_args()

    with open(args.suite) as f:
        cases = json.load(f)

    completed = load_completed(args.output)

    # Write header only if file is new
    if not os.path.exists(args.output):
        write_header(args.output)

    pending = [c for c in cases if c["case_id"] not in completed]
    total = len(cases)
    done = len(completed)

    print(f"Suite: {args.suite}")
    print(f"Output: {args.output}")
    print(f"Cases: {total} total, {done} already done, {len(pending)} to run")
    print(f"Timeout: {args.timeout}s, RAM cap: {args.mem_limit_mb}MB\n")

    for i, case in enumerate(pending):
        elapsed_label = f"[{done + i + 1}/{total}]"
        print(f"{elapsed_label} {case['case_id']} ... ", end="", flush=True)

        t_wall = time.perf_counter()
        row = run_case(case, args.suite, args.timeout, args.mem_limit_mb)
        wall = time.perf_counter() - t_wall

        append_row(args.output, row)
        status = row["status"]
        print(f"{status} ({wall:.1f}s)")

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Run against smoke suite with 3-case subset**

Create a tiny test manifest:
```bash
python -c "
import json
cases = json.load(open('bench/suites/smoke.json'))
json.dump(cases[:3], open('bench/suites/tiny.json', 'w'), indent=2)
print('Wrote 3 cases to bench/suites/tiny.json')
"
```

**Step 3: Run the runner**

```bash
python bench/run_bench.py \
    --suite bench/suites/tiny.json \
    --output bench/results/tiny_results.csv \
    --timeout 120 \
    --mem-limit-mb 2048
```
Expected: 3 lines printed (`[1/3]`, `[2/3]`, `[3/3]`), each with `ok` or a known status.

**Step 4: Verify CSV**

```bash
python -c "
import pandas as pd
df = pd.read_csv('bench/results/tiny_results.csv')
print(df.to_string())
"
```
Expected: 3 rows with all CSV fields, no parse errors.

**Step 5: Test resumability** — re-run the same command and confirm 0 new cases are run:

```bash
python bench/run_bench.py \
    --suite bench/suites/tiny.json \
    --output bench/results/tiny_results.csv \
    --timeout 120 \
    --mem-limit-mb 2048
```
Expected: `3 total, 3 already done, 0 to run`.

**Step 6: Commit**

```bash
git add bench/run_bench.py bench/suites/tiny.json bench/results/tiny_results.csv
git commit -m "#add: bench/run_bench.py - subprocess runner with timeout, RAM cap, and resumability"
```

---

## Task 5: `bench/analyze.ipynb`

**Files:**
- Create: `bench/analyze.ipynb`

**Background:** The notebook reads the results CSV and produces the six plots and summary table defined in the design doc. It uses `matplotlib` + `seaborn`. All cells should be runnable top-to-bottom without errors (on a populated results CSV).

**Step 1: Create the notebook**

Create `bench/analyze.ipynb` with the following cells (JSON format below — write via Jupyter or create as `.py` and convert):

**Cell 1 — Imports and load data:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="colorblind")
RESULTS_CSV = "results/tiny_results.csv"  # Update to full results path

df = pd.read_csv(RESULTS_CSV)
ok = df[df["status"] == "ok"].copy()
print(f"Total rows: {len(df)}, ok: {len(ok)}, non-ok: {len(df) - len(ok)}")
```

**Cell 2 — Plot 1: Scaling vs. n:**
```python
subset = ok[(ok["num_aps"] == 3) & (ok["formula_size"] == 5) & (ok["density"] == "medium")]
agg = subset.groupby("n").agg(
    t_mean=("t_total", "mean"), t_std=("t_total", "std"),
    states_mean=("semi_states", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.errorbar(agg["n"], agg["t_mean"], yerr=agg["t_std"], marker="o", label="t_total (s)")
ax1.set_xlabel("Number of formulas (n)")
ax1.set_ylabel("Translation time (s)")
ax2 = ax1.twinx()
ax2.plot(agg["n"], agg["states_mean"], marker="s", linestyle="--", color="orange", label="semi_states")
ax2.set_ylabel("Semi-automaton states")
ax1.set_title("Scaling vs. number of formulas")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
plt.savefig("results/plot_scaling_n.pdf", bbox_inches="tight")
plt.show()
```

**Cell 3 — Plot 2: Scaling vs. |AP|:**
```python
subset = ok[(ok["n"] == 4) & (ok["formula_size"] == 5) & (ok["density"] == "medium")]
agg = subset.groupby("num_aps").agg(
    t_mean=("t_total", "mean"), t_std=("t_total", "std"),
    states_mean=("semi_states", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.errorbar(agg["num_aps"], agg["t_mean"], yerr=agg["t_std"], marker="o")
ax1.set_xlabel("Atomic propositions (|AP|)")
ax1.set_ylabel("Translation time (s)")
ax2 = ax1.twinx()
ax2.plot(agg["num_aps"], agg["states_mean"], marker="s", linestyle="--", color="orange")
ax2.set_ylabel("Semi-automaton states")
ax1.set_title("Scaling vs. number of atomic propositions")
plt.tight_layout()
plt.savefig("results/plot_scaling_aps.pdf", bbox_inches="tight")
plt.show()
```

**Cell 4 — Plot 3: Scaling vs. formula_size:**
```python
subset = ok[(ok["n"] == 4) & (ok["num_aps"] == 3) & (ok["density"] == "medium")]
agg = subset.groupby("formula_size").agg(
    t_mean=("t_total", "mean"), t_std=("t_total", "std"),
    states_mean=("semi_states", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.errorbar(agg["formula_size"], agg["t_mean"], yerr=agg["t_std"], marker="o")
ax1.set_xlabel("Formula size (max operators)")
ax1.set_ylabel("Translation time (s)")
ax2 = ax1.twinx()
ax2.plot(agg["formula_size"], agg["states_mean"], marker="s", linestyle="--", color="orange")
ax2.set_ylabel("Semi-automaton states")
ax1.set_title("Scaling vs. formula size")
plt.tight_layout()
plt.savefig("results/plot_scaling_fsize.pdf", bbox_inches="tight")
plt.show()
```

**Cell 5 — Plot 4: Density comparison (grouped bar):**
```python
subset = ok[(ok["n"] == 4) & (ok["num_aps"] == 3) & (ok["formula_size"] == 5)]
agg = subset.groupby("density").agg(
    t_mean=("t_total", "mean"),
    states_mean=("semi_states", "mean"),
    pref_edges_mean=("pref_edges", "mean"),
).reset_index()

density_order = ["sparse", "medium", "dense", "total"]
agg["density"] = pd.Categorical(agg["density"], categories=density_order, ordered=True)
agg = agg.sort_values("density")

x = np.arange(len(agg))
width = 0.25
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width, agg["t_mean"], width, label="t_total (s)")
ax.bar(x, agg["states_mean"] / agg["states_mean"].max(), width, label="semi_states (norm)")
ax.bar(x + width, agg["pref_edges_mean"] / (agg["pref_edges_mean"].max() + 1e-9), width, label="pref_edges (norm)")
ax.set_xticks(x)
ax.set_xticklabels(agg["density"])
ax.set_xlabel("Partial order density")
ax.set_title("Effect of partial order density")
ax.legend()
plt.tight_layout()
plt.savefig("results/plot_density.pdf", bbox_inches="tight")
plt.show()
```

**Cell 6 — Plot 5: Stage breakdown (stacked bar):**
```python
subset = ok[(ok["num_aps"] == 3) & (ok["formula_size"] == 5) & (ok["density"] == "medium")]
agg = subset.groupby("n").agg(
    t_dfa=("t_dfa", "mean"), t_semi=("t_semi", "mean"), t_pref=("t_pref", "mean")
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(agg["n"], agg["t_dfa"], label="LTLf→DFA (t_dfa)")
ax.bar(agg["n"], agg["t_semi"], bottom=agg["t_dfa"], label="Semi-automaton (t_semi)")
ax.bar(agg["n"], agg["t_pref"], bottom=agg["t_dfa"] + agg["t_semi"], label="Pref graph (t_pref)")
ax.set_xlabel("Number of formulas (n)")
ax.set_ylabel("Time (s)")
ax.set_title("Pipeline stage breakdown vs. n")
ax.legend()
plt.tight_layout()
plt.savefig("results/plot_stage_breakdown.pdf", bbox_inches="tight")
plt.show()
```

**Cell 7 — Plot 6: Feasibility heatmap:**
```python
heat = df.groupby(["n", "num_aps"]).apply(
    lambda g: (g["status"] == "ok").mean()
).reset_index(name="success_rate")
pivot = heat.pivot(index="n", columns="num_aps", values="success_rate")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
ax.set_title("Success rate: n × |AP| (formula_size=5, density=medium)")
ax.set_xlabel("|AP|")
ax.set_ylabel("n")
plt.tight_layout()
plt.savefig("results/plot_heatmap.pdf", bbox_inches="tight")
plt.show()
```

**Cell 8 — Summary table:**
```python
summary = ok.groupby(["n", "num_aps", "formula_size", "density"]).agg(
    mean_t_total=("t_total", "mean"),
    mean_semi_states=("semi_states", "mean"),
    mean_pref_nodes=("pref_nodes", "mean"),
).round(3).reset_index()

total_per_combo = df.groupby(["n", "num_aps", "formula_size", "density"]).size().reset_index(name="total")
ok_per_combo = ok.groupby(["n", "num_aps", "formula_size", "density"]).size().reset_index(name="ok_count")
rate = total_per_combo.merge(ok_per_combo, how="left").fillna(0)
rate["success_pct"] = (rate["ok_count"] / rate["total"] * 100).round(1)

summary = summary.merge(rate[["n", "num_aps", "formula_size", "density", "success_pct"]], how="left")
print(summary.to_string(index=False))
summary.to_csv("results/summary_table.csv", index=False)
```

**Step 2: Verify notebook structure**

Open `bench/analyze.ipynb` in Jupyter and run Cell 1 against `tiny_results.csv`. Expected: no import errors, DataFrame loads.

**Step 3: Commit**

```bash
git add bench/analyze.ipynb
git commit -m "#add: bench/analyze.ipynb - analysis notebook with scaling plots and summary table"
```

---

## Task 6: `bench/README.md`

**Files:**
- Create: `bench/README.md`

**Step 1: Write README**

```markdown
# PrefLTLf → PDFA Scalability Benchmark

Scalability benchmark for `PrefLTLf.translate()`. Tests translation time, memory,
and output size across number of formulas, atomic propositions, formula complexity,
and partial order density.

## Platform Requirement

**Run on WSL2 (Ubuntu) inside Windows 11.** The RAM-capping mechanism uses
`resource.setrlimit(RLIMIT_AS)`, a POSIX-only API. Do not run from native
Windows shells (cmd, PowerShell, Git Bash).

## Quickstart

```bash
# 1. Generate the benchmark manifest
python bench/gen_suite.py --output bench/suites/suite.json

# 2. Run (adjust timeout and RAM cap as needed)
python bench/run_bench.py \
    --suite bench/suites/suite.json \
    --output bench/results/results.csv \
    --timeout 300 \
    --mem-limit-mb 4096

# 3. Analyze — open bench/analyze.ipynb in Jupyter
#    Update RESULTS_CSV in Cell 1 to point to your results CSV.
jupyter notebook bench/analyze.ipynb
```

## Resuming an interrupted run

Re-run step 2 with the same `--output` path. Cases already in the CSV are skipped.

## Output CSV columns

| Column | Meaning |
|--------|---------|
| `case_id` | Unique case identifier |
| `n` | Number of LTLf formulas |
| `num_aps` | Atomic propositions |
| `formula_size` | Max operator count |
| `density` | Partial order density label |
| `seed` | Random seed |
| `status` | `ok` / `timeout` / `oom` / `error` |
| `t_dfa` | LTLf→DFA stage time (s) |
| `t_semi` | Semi-automaton stage time (s) |
| `t_pref` | Preference graph stage time (s) |
| `t_total` | Total translation time (s) |
| `peak_mem_mb` | Peak memory (MB, tracemalloc) |
| `semi_states` | States in semi-automaton |
| `semi_transitions` | Transitions in semi-automaton |
| `pref_nodes` | Nodes in preference graph |
| `pref_edges` | Edges in preference graph |

## Defaults

- Timeout: 300 s
- RAM cap: 4096 MB
- Seeds: 0–4 (5 repetitions per parameter combination)
- ~300 total cases
```

**Step 2: Commit**

```bash
git add bench/README.md
git commit -m "#add: bench/README.md - platform note, quickstart, CSV schema"
```

---

## Task 7: Full suite smoke-test

**Step 1: Generate full suite**

```bash
python bench/gen_suite.py --output bench/suites/suite_full.json
python -c "import json; print(len(json.load(open('bench/suites/suite_full.json'))), 'cases')"
```
Expected: ~300 cases.

**Step 2: Run 10-case subset as final integration check**

```bash
python -c "
import json
cases = json.load(open('bench/suites/suite_full.json'))
# Pick cases with n=2 (fastest) and seed=0
small = [c for c in cases if c['n'] == 2 and c['seed'] == 0]
json.dump(small, open('bench/suites/suite_small.json', 'w'), indent=2)
print(len(small), 'cases written')
"

python bench/run_bench.py \
    --suite bench/suites/suite_small.json \
    --output bench/results/small_results.csv \
    --timeout 120 \
    --mem-limit-mb 2048
```
Expected: all cases complete with `ok` or known status, no Python tracebacks.

**Step 3: Verify CSV is readable by the notebook**

```bash
python -c "
import pandas as pd
df = pd.read_csv('bench/results/small_results.csv')
print(df[['case_id','status','t_total','semi_states','pref_nodes']].to_string())
"
```

**Step 4: Final commit**

```bash
git add bench/suites/suite_small.json bench/results/small_results.csv
git commit -m "#add: smoke-test suite and results for bench/ integration check"
```
