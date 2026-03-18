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


def run_case(case: dict, manifest_path: str, timeout: int, mem_limit_mb: int, python: str = None) -> dict:
    """Spawn worker.py for one case; return result dict."""
    worker = Path(__file__).parent / "worker.py"
    cmd = [
        python or sys.executable, str(worker),
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

    # Try to parse worker JSON output — take the last line in case of debug output
    if stdout:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    metrics = json.loads(line)
                    if metrics.get("status") == "ok":
                        return {**base, **metrics}
                    else:
                        status = metrics.get("status", "error")
                        return {**base, "status": status}
                except json.JSONDecodeError:
                    continue

    # Worker printed nothing parseable — check exit code
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
    parser.add_argument("--python", default=None,
                        help="Python executable for worker subprocesses (default: sys.executable). "
                             "Use this when the runner Python differs from the one with spot/prefltlf2pdfa installed, "
                             "e.g. --python /usr/bin/python3")
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
    print(f"Timeout: {args.timeout}s, RAM cap: {args.mem_limit_mb}MB")
    print(f"Worker Python: {args.python or sys.executable}\n")

    for i, case in enumerate(pending):
        elapsed_label = f"[{done + i + 1}/{total}]"
        print(f"{elapsed_label} {case['case_id']} ... ", end="", flush=True)

        t_wall = time.perf_counter()
        row = run_case(case, args.suite, args.timeout, args.mem_limit_mb, python=args.python)
        wall = time.perf_counter() - t_wall

        append_row(args.output, row)
        status = row["status"]
        print(f"{status} ({wall:.1f}s)")

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
