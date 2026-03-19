# Benchmark Design: PrefLTLf → PDFA Scalability

**Date:** 2026-03-18
**Goal:** Stress-test the scalability limits of `PrefLTLf.translate()` and produce paper-quality comparison results across structural parameters and partial order density.

---

## Platform Requirement

> **This benchmark is designed to run on Ubuntu/WSL (Windows Subsystem for Linux).**
> The RAM-capping mechanism uses `resource.setrlimit(resource.RLIMIT_AS, ...)`, which is a POSIX-only API available in WSL but not in native Windows.
> The recommended setup is **Windows 11 + WSL2** with the project installed inside the WSL environment.
> Do not run `run_bench.py` from a native Windows shell (cmd, PowerShell, Git Bash) — timeout and memory limits will not behave correctly.

---

## Directory Layout

```
bench/
├── gen_suite.py          # generates benchmark manifest (JSON)
├── run_bench.py          # runner: spawns each case in subprocess, writes results CSV
├── worker.py             # subprocess entry point: runs one case, prints metrics as JSON
├── analyze.ipynb         # analysis notebook: plots + tables for paper
├── suites/               # generated manifests (gitignored)
│   └── suite_<timestamp>.json
├── results/              # raw results CSVs (gitignored)
│   └── results_<timestamp>.csv
└── README.md             # platform note, quickstart
```

`suites/` and `results/` are gitignored (large generated data).

---

## File Responsibilities

### `gen_suite.py`
Pure data generation — no translation occurs here. Outputs a JSON list of case dicts. Each case contains all parameters needed to fully reproduce a run, including the pre-generated formulas and partial order edges.

CLI:
```
python gen_suite.py --output suites/suite_<timestamp>.json
```

### `run_bench.py`
Reads a manifest, iterates over cases, spawns `worker.py` per case as a subprocess with configurable timeout and RAM limit. Captures stdout/stderr, appends one result row to a CSV. Supports resuming (skips `case_id`s already present in the output CSV).

CLI:
```
python run_bench.py \
    --suite suites/suite.json \
    --output results/results.csv \
    --timeout 300 \
    --mem-limit-mb 4096
```

### `worker.py`
The only file that imports `prefltlf2pdfa`. Receives a manifest path and case ID. At startup, applies `resource.setrlimit` to cap its own RAM. Instruments each pipeline stage with `time.perf_counter()` and wraps the full `translate()` call with `tracemalloc`. Prints a single JSON line to stdout containing all metrics, then exits.

CLI (invoked by runner, not directly):
```
python worker.py \
    --manifest suites/suite.json \
    --case-id n4_ap3_sz5_dense_seed2 \
    --mem-limit-mb 4096
```

### `analyze.ipynb`
Reads the results CSV into pandas. Produces all plots and the summary table described in the Analysis section below.

---

## Parameter Space

### Axis 1 — Structural parameters

Each parameter is swept independently with others held at the baseline value.

| Parameter | Range | Baseline |
|-----------|-------|----------|
| `n` — number of LTLf formulas | 2, 3, 4, 5, 6, 8, 10 | 4 |
| `num_aps` — atomic propositions | 2, 3, 4, 5 | 3 |
| `formula_size` — max operator count (passed to `spot.randltl`) | 3, 5, 8, 12 | 5 |

### Axis 2 — Partial order density

| Level | Edge probability `p` | Label |
|-------|---------------------|-------|
| Sparse | 0.2 | `sparse` |
| Medium | 0.5 | `medium` |
| Dense | 0.8 | `dense` |
| Total preorder | 1.0 | `total` |

**Repetitions:** 5 random seeds per combination, for statistical confidence on randomly generated formulas and partial orders.

**Approximate total cases:**
- Axis 1 sweep: (7 + 4 + 4) × 4 density levels × 5 seeds = 300 cases
- Axis 2 sweep at baseline structural params: 4 × 5 = 20 cases (already included above)
- Grand total: ~300 cases

### Manifest Schema

One JSON object per case:
```json
{
  "case_id": "n4_ap3_sz5_dense_seed2",
  "n": 4,
  "num_aps": 3,
  "formula_size": 5,
  "density": "dense",
  "density_p": 0.8,
  "seed": 2,
  "formulas": ["F a", "G b", "F c", "!(F(a) | G(b))"],
  "partial_order": [[0, 1], [0, 2], [1, 3]]
}
```

Formulas and the partial order are pre-generated and embedded in the manifest so that every run is fully reproducible without re-invoking `spot.randltl`.

---

## Metrics

### Per-stage timing

`worker.py` records wall-clock time around each pipeline stage:

```
t0 → [LTLf→DFA via MONA, one call per formula]  → t1   (t_dfa   = t1 - t0)
t1 → [semi-automaton (product) construction]      → t2   (t_semi  = t2 - t1)
t2 → [preference graph construction]              → t3   (t_pref  = t3 - t2)
                                                          (t_total = t3 - t0)
```

### Memory

`tracemalloc` wraps the full `translate()` call. Peak memory is recorded in MB.

### Output size

Extracted from the returned `PrefAutomaton`:
- `semi_states` — number of states in the semi-automaton
- `semi_transitions` — number of transitions in the semi-automaton
- `pref_nodes` — nodes in the preference graph (equivalence classes)
- `pref_edges` — edges in the preference graph

### Timeout and failure

`run_bench.py` enforces a subprocess-level timeout (`subprocess.run(..., timeout=T)`). `worker.py` enforces a RAM cap via `resource.setrlimit` at startup. Result `status` values:

| Status | Meaning |
|--------|---------|
| `ok` | Translation completed within time and memory limits |
| `timeout` | Subprocess exceeded wall-clock timeout |
| `oom` | Worker process was killed by OS due to RAM cap |
| `error` | Worker exited with non-zero code (Python exception) |

### Result CSV Schema

```
case_id, n, num_aps, formula_size, density, seed,
status, t_dfa, t_semi, t_pref, t_total,
peak_mem_mb, semi_states, semi_transitions,
pref_nodes, pref_edges
```

Rows with `status != ok` have empty metric columns.

---

## Analysis

The `analyze.ipynb` notebook produces the following outputs from the results CSV.

### Plots

1. **Scaling vs. `n`** — line plot of `t_total` (mean ± std across seeds) vs. number of formulas; secondary y-axis for `semi_states`. Fixed: `|AP|=3`, `formula_size=5`, `density=medium`.
2. **Scaling vs. `|AP|`** — same style. Fixed: `n=4`, `formula_size=5`, `density=medium`.
3. **Scaling vs. `formula_size`** — same style. Fixed: `n=4`, `|AP|=3`, `density=medium`.
4. **Density comparison** — grouped bar chart across 4 density levels for `t_total`, `semi_states`, `pref_edges`. Fixed: `n=4`, `|AP|=3`, `formula_size=5`.
5. **Stage breakdown** — stacked bar of `t_dfa` / `t_semi` / `t_pref` across the `n` sweep; shows which stage dominates at each scale.
6. **Feasibility heatmap** — `n` × `|AP|` grid colored by success rate (fraction of seeds with `status=ok`); identifies the feasibility boundary.

### Summary Table

One row per parameter combination: mean `t_total`, mean `semi_states`, mean `pref_nodes`, success rate (%).

---

## Formula Generation

Formulas are generated using `spot.randltl(num_aps, n, seed=seed)` with `.simplify().unabbreviate("WMR")`, consistent with `examples/formula_generator.py`. The `formula_size` parameter controls the operator budget passed to `randltl`.

Partial orders are generated using the existing `generate_random_partial_order(n, p)` logic from `formula_generator.py`, extended to accept a density probability `p` instead of a fixed 0.5.

---

## Resumability

`run_bench.py` checks the output CSV at startup and skips any `case_id` already present. Interrupted runs can be resumed by re-running the same command.
