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

# If running from a conda env that doesn't have spot, point --python at the
# Python where spot and prefltlf2pdfa are installed (e.g. system python3):
python bench/run_bench.py \
    --suite bench/suites/suite.json \
    --output bench/results/results.csv \
    --timeout 300 \
    --mem-limit-mb 4096 \
    --python /usr/bin/python3

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
| `peak_mem_mb` | Peak Python-heap memory (MB, tracemalloc) |
| `max_rss_mb` | Peak OS resident set size (MB, `resource.getrusage`, Linux/WSL only) |
| `semi_states` | States in semi-automaton |
| `semi_transitions` | Transitions in semi-automaton |
| `pref_nodes` | Nodes in preference graph |
| `pref_edges` | Edges in preference graph |

## Defaults

- Timeout: 300 s
- RAM cap: 4096 MB
- Seeds: 0–4 (5 repetitions per parameter combination)
- ~300 total cases (expanded parameter space: n up to 30, |AP| up to 10, formula_size up to 20)
