"""
run_bench.py — Run benchmark suite, one subprocess per case.

Usage:
    python run_bench.py \
        --suite suites/suite.json \
        --output results/results.csv \
        --timeout 300 \
        --mem-limit-mb 4096 \
        --workers 3

Supports resuming: skips case_ids already present in --output CSV.
Log file defaults to <output>.log (e.g. results/results.log).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

CSV_FIELDS = [
    "case_id", "n", "num_aps", "formula_size", "density", "seed",
    "status", "t_dfa", "t_semi", "t_pref", "t_total",
    "peak_mem_mb", "max_rss_mb", "semi_states", "semi_transitions", "pref_nodes", "pref_edges",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_path: str):
    """Configure loguru: INFO+ to stderr (no colour), DEBUG+ to log file."""
    logger.remove()  # remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level: <8}</level> | {message}",
        colorize=False,
    )
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="50 MB",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_path}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

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
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Worker runner
# ---------------------------------------------------------------------------

def _first_line(text: str) -> str:
    """Return the first non-empty line of text, truncated to 120 chars."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return ""


def run_case(case: dict, manifest_path: str, timeout: int, mem_limit_mb: int, python: str = None) -> dict:
    """Spawn worker.py for one case; return result dict (includes non-CSV 'error' key on failure)."""
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

    logger.debug(f"Spawning worker | case_id={case['case_id']} cmd={' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"Exceeded {timeout}s timeout"
        logger.warning(f"TIMEOUT | {case['case_id']} | {msg}")
        return {**base, "status": "timeout", "error": msg}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    logger.debug(f"Worker finished | case_id={case['case_id']} returncode={result.returncode}")
    if stderr:
        logger.debug(f"Worker stderr | {case['case_id']}:\n{stderr}")

    # Try to parse worker JSON output — take the last JSON line in case of debug output
    if stdout:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    metrics = json.loads(line)
                    status = metrics.get("status", "error")
                    if status == "ok":
                        logger.debug(f"OK | {case['case_id']} | t_total={metrics.get('t_total')}s")
                        return {**base, **metrics}
                    else:
                        error_msg = metrics.get("error", "") or stderr
                        if status == "oom":
                            logger.warning(f"OOM  | {case['case_id']} | {_first_line(error_msg)}")
                        else:
                            logger.error(
                                f"ERROR | {case['case_id']} | {_first_line(error_msg)}\n"
                                f"Full traceback:\n{error_msg}"
                            )
                        return {**base, "status": status, "error": error_msg}
                except json.JSONDecodeError:
                    continue

    # Worker printed nothing parseable — check exit code / stderr
    if result.returncode in (-9, -11):
        msg = f"Killed by OS signal {result.returncode} (OOM via RLIMIT_AS)"
        logger.warning(f"OOM  | {case['case_id']} | {msg}")
        return {**base, "status": "oom", "error": msg}

    if result.returncode != 0:
        error_msg = stderr or stdout or f"Worker exited with code {result.returncode}"
        logger.error(
            f"ERROR | {case['case_id']} | exit={result.returncode} | {_first_line(error_msg)}\n"
            f"Full output:\n{error_msg}"
        )
        return {**base, "status": "error", "error": error_msg}

    error_msg = stderr or "Worker exited 0 but produced no parseable output"
    logger.error(f"ERROR | {case['case_id']} | {_first_line(error_msg)}")
    return {**base, "status": "error", "error": error_msg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="Path to manifest JSON.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case timeout in seconds.")
    parser.add_argument("--mem-limit-mb", type=int, default=4096, help="RAM cap per worker in MB.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of cases to run in parallel (default: 1). "
                             "Each worker is an independent subprocess; --mem-limit-mb applies per worker.")
    parser.add_argument("--python", default=None,
                        help="Python executable for worker subprocesses (default: sys.executable). "
                             "Use this when the runner Python differs from the one with spot/prefltlf2pdfa installed, "
                             "e.g. --python /usr/bin/python3")
    parser.add_argument("--log-file", default=None,
                        help="Path to log file (default: <output>.log, e.g. results/results.log).")
    args = parser.parse_args()

    log_path = args.log_file or str(Path(args.output).with_suffix(".log"))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    setup_logger(log_path)

    with open(args.suite) as f:
        cases = json.load(f)

    completed = load_completed(args.output)

    # Write header only if file is new
    if not os.path.exists(args.output):
        write_header(args.output)

    pending = [c for c in cases if c["case_id"] not in completed]
    total = len(cases)
    done_count = len(completed)

    logger.info(f"Suite:         {args.suite}")
    logger.info(f"Output:        {args.output}")
    logger.info(f"Cases:         {total} total, {done_count} already done, {len(pending)} to run")
    logger.info(f"Timeout:       {args.timeout}s  |  RAM cap: {args.mem_limit_mb}MB/worker")
    logger.info(f"Workers:       {args.workers} parallel  |  total RAM budget: ~{args.workers * args.mem_limit_mb / 1024:.1f}GB")
    logger.info(f"Worker Python: {args.python or sys.executable}")
    print()

    # Shared mutable state accessed from multiple threads
    finished = done_count          # how many cases are fully done (for progress label)
    csv_lock = threading.Lock()    # serialises CSV writes and progress prints

    def _run_and_record(case: dict) -> None:
        nonlocal finished

        t_wall = time.perf_counter()
        row = run_case(case, args.suite, args.timeout, args.mem_limit_mb, python=args.python)
        wall = time.perf_counter() - t_wall

        with csv_lock:
            finished += 1
            label = f"[{finished}/{total}]"
            append_row(args.output, row)

            status = row["status"]
            if status == "ok":
                print(f"{label} {case['case_id']} ... ok ({wall:.1f}s)")
            else:
                short = _first_line(row.get("error", "") or "")
                suffix = f" — {short}" if short else ""
                print(f"{label} {case['case_id']} ... {status} ({wall:.1f}s){suffix}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_and_record, case): case for case in pending}
        for future in as_completed(futures):
            # Re-raise any unexpected exception from the thread itself
            # (worker errors are caught inside run_case; this guards against bugs here)
            exc = future.exception()
            if exc:
                logger.error(f"Unexpected thread error for {futures[future]['case_id']}: {exc}")

    logger.info(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
