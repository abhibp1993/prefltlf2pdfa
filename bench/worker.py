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
    pref = PrefLTLf(spec_str, auto_complete="incomparable")

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
        enumeration="alphabet-enumeration",
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
