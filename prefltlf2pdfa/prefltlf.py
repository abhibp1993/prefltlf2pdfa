"""
Module for handling PrefLTLf formulas and converting them to preference automata.
PrefLTLf formulas enable expressing preferences over LTLf formulas.

This module utilizes the `ltlf2dfa` library to parse and process LTLf formulas
into preference deterministic finite automata (PDFA). It also provides functions for
validating and manipulating preference automata.

.. note:: The module checks for the presence of the `mona` tool, which is essential
    for translating LTLf to PDFA. If `mona` is not found, a warning message is printed
    to inform the user. In this case, the `translate` functionality may not work properly,
    but the remaining functions work okay.
"""

try:
    import spot
except ImportError:
    spot = None

try:
    import sympy
except ImportError:
    sympy = None

import itertools
import os
import sys

import math

import lark.exceptions
import networkx as nx
import pprint
import subprocess
from jinja2.idtracking import symbols_for_node

from loguru import logger
from ltlf2dfa.parser.ltlf import LTLfParser
# from sympy.abc import alpha
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pqdm.processes import pqdm

import prefltlf2pdfa.utils as utils
from prefltlf2pdfa.semantics import *

# logger.remove()
PARSER = LTLfParser()

# Check ltlf2dfa is functioning properly and has access to mona.
command = "mona --help"

# Run the command using subprocess
try:
    result = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "Usage: mona [options] <filename>" not in result.stdout:
        YELLOW = '\033[33m'
        RESET = '\033[0m'
        print(f"{YELLOW}Mona not found on your system. The tool may not translate LTLf to PDFA properly.{RESET}")
        logger.warning("Mona not found on your system. The tool may not translate LTLf to PDFA properly.")

        # Output and return code
        logger.debug("Return Code:", result.returncode)
        logger.debug("Output:", result.stdout)
        logger.debug("Error (if any):", result.stderr)

except Exception as e:
    print(f"An error occurred: {e}")


class PrefLTLf:
    """
    Represents a PrefLTLf (Preference Linear Temporal Logic over Finite Traces) formula.

    The class is used to define, parse, and serialize PrefLTLf specifications,
    as well as to translate the formulas into automata.

    Attributes
    ----------
    raw_spec : str
        The raw specification of the PrefLTLf formula.

    atoms : set
        Set of atomic propositions appearing in the PrefLTLf specification.

    alphabet : list
        A list representing the alphabet for the specification, or `None` if not provided.

    phi : dict
        A dictionary of LTLf formulas appearing in the PrefLTLf specification.

    relation : set
        A set of triples (PREF_TYPE, LTLf, LTLf) constructed based on the given PrefLTLf specification.

    """
    MAXIMAL_SEMANTICS = [semantics_mp_forall_exists, semantics_mp_exists_forall, semantics_mp_forall_forall]

    def __init__(self, spec, alphabet=None, **kwargs):
        """
        Initializes the PrefLTLf formula.

        Parameters
        ----------
        spec : str
            The raw specification of the PrefLTLf formula.

        alphabet : list, optional
            The alphabet for the specification (default is None).

        **kwargs
            Additional options for parsing and handling the formula.

            - skip_parse : bool, optional
                If True, skips the parsing of the formula during initialization. Default is False.

            - auto_complete : str, optional
                Specifies how the formula should handle incomplete preferences when parsing. The acceptable options are:

                * "none": No auto-completion. The formula must be fully specified, and any incompleteness will raise an error.
                * "incomparable": If preferences are incomplete, auto-completes them by marking certain conditions as incomparable.
                * "minimal": Auto-completes the preferences by adding the minimal number of relations needed to make the preferences complete.

                Default is "none".
        """
        # Class state variables (will be serialized)
        self.raw_spec = spec
        self.atoms = set()  # Set of atomic propositions appearing in PrefLTLf specification
        self.alphabet = list(alphabet) if alphabet is not None else None
        self.phi = dict()  # Dict (indexed set) of LTLf formulas appearing in PrefLTLf specification
        self.relation = set()  # Set of triples (PREF_TYPE, LTLf, LTLf) constructed based on given PrefLTLf spec
        # self.dfa = list()  # List (indexed set) of LTLf formulas appearing in PrefLTLf specification

        if not kwargs.get("skip_parse", False):
            self.parse(
                auto_complete=kwargs.get("auto_complete", "none"),
            )

    def __eq__(self, other):
        keys = ["f_str", "atoms", "alphabet", "phi", "relation"]
        if isinstance(other, PrefLTLf):
            return ({k: v for k, v in self.serialize().items() if k in keys} ==
                    {k: v for k, v in other.serialize().items() if k in keys})
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.relation)))

    def __str__(self):
        return pprint.pformat(self.serialize())

    def __repr__(self):
        return f"<PrefLTLf Formula at {id(self)}>"

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        self.raw_spec = obj_dict["f_str"]
        self.atoms = set(obj_dict["atoms"])
        self.alphabet = [set(p) for p in obj_dict["alphabet"]] if obj_dict["alphabet"] is not None else None
        self.phi = obj_dict["phi"]
        # self.dfa = obj_dict["dfa"]
        self.relation = set(obj_dict["relation"])
        self.parse(auto_complete="none")

    def serialize(self):
        """
        Serializes the PrefLTLf formula into a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary representing the serialized PrefLTLf formula.
        """
        jsonable_dict = {
            "f_str": self.raw_spec,
            "atoms": list(self.atoms),
            "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "phi": {k: str(v) for k, v in self.phi.items()},
            "relation": list(self.relation),
            # "dfa": self.dfa
        }
        return jsonable_dict

    @classmethod
    def deserialize(cls, obj_dict):
        """
        Creates a PrefLTLf formula from a serialized dictionary.

        Parameters
        ----------
        obj_dict : dict
            A dictionary representing the serialized PrefLTLf formula.

        Returns
        -------
        PrefLTLf
            A new instance of the PrefLTLf formula.
        """
        formula = cls(spec=obj_dict["f_str"], skip_parse=True)
        formula.atoms = set(obj_dict["atoms"])
        formula.alphabet = [set(p) for p in obj_dict["alphabet"]] if obj_dict["alphabet"] is not None else None
        formula.phi = {int(k): PARSER(v) for k, v in obj_dict["phi"].items()}
        formula.relation = set(obj_dict["relation"])
        # formula.dfa = obj_dict["dfa"]
        return formula

    @classmethod
    def from_file(cls, fpath, alphabet=None, auto_complete="none"):
        """
        Reads a PrefLTLf formula from a file.

        Parameters
        ----------
        fpath : str
            The file path to read the formula from.

        alphabet : list, optional
            The alphabet for the specification (default is None).

        auto_complete : str, optional
            The auto-completion method to use (default is "none").

        Returns
        -------
        PrefLTLf
            A new instance of the PrefLTLf formula.
        """
        with open(fpath, 'r') as f:
            return cls(f.read(), alphabet=alphabet, auto_complete=auto_complete)

    def parse(self, auto_complete="none"):
        """
        Parses the raw PrefLTLf formula and constructs the atomic propositions, relation, and LTLf formulas.

        Parameters
        ----------
        auto_complete : str, optional
            The method to use for auto-completion (default is "none").

        Returns
        -------
        tuple
            A tuple containing
            - phi: Dictionary of {formula-id: LTLf Formula}
            - model: Set of binary relation of form (a, b) representing a is weakly preferred to b.
            - atoms: Set of strings representing atoms.
        """
        logger.debug(f"Parsing prefltlf formula with {auto_complete=}: \n{self.raw_spec}")

        # Check auto_complete inputs
        if auto_complete not in ["none", "incomparable", "minimal"]:
            logger.error(
                f"Unknown auto_complete value '{auto_complete}' to parse function. "
                f"Accepted values are {['none', 'incomparable', 'minimal']}. "
                f"New auto_complete value is set to 'none'."
            )
            auto_complete = "none"

        # # Parse header. Ensure that the spec is well-formed and a PrefLTLf formula.
        # header = self._parse_header()

        # Construct intermediate representation of standard spec
        phi, spec_ir = self._construct_spec_ir()
        if not self._is_lang_complete(phi):
            if auto_complete != "none":
                phi, spec_ir = self._auto_complete(phi, spec_ir, auto_complete)
            else:
                logger.error(f"The set of conditions phi = {set(phi.values())} is not complete")
                raise ValueError(f"The set of conditions phi = {set(phi.values())} is not complete")

        # Else, we have spec in standard format. Build model. (if user has specified ltlf formulas, add them to model)
        phi, model = self._build_partial_order(phi, spec_ir)
        # print(f"{phi=}")
        # print(f"{model=}")

        # Construct set of atomic propositions
        atoms = set()
        for varphi in phi.values():
            atoms.update(set(varphi.find_labels()))

        # Return
        self.phi = phi
        self.relation = model
        self.atoms = atoms

        return phi, model, atoms

    def translate(self, semantics=semantics_mp_forall_exists, show_progress=False):
        """
        Translates the PrefLTLf formula into a preference automaton under the given semantics.

        Parameters
        ----------
        semantics : function, optional
            The semantics function to use for translation (default is `semantics_mp_forall_exists`).

        show_progress : bool, optional
            Whether to show progress during translation (default is False).

        Returns
        -------
        PrefAutomaton
            The resulting preference automaton.
        """
        logger.debug(
            f"Translating the formula to preference automaton under semantics=`{semantics.__name__}`:\n{self.raw_spec}"
        )

        # Define preference automaton and set basic attributes
        aut = PrefAutomaton()
        aut.atoms = self.atoms
        aut.alphabet = self.alphabet
        aut.phi = self.phi

        # Translate LTLf formulas in self.phi to DFAs
        aut.sorted_phi = sorted(self.phi.keys())
        aut.dfa = [utils.ltlf2dfa(self.phi[i]) for i in aut.sorted_phi]
        assert len(aut.dfa) >= 2, f"PrefLTLf spec must have at least two LTLf formulas."

        # Log all DFAs
        log_message = f"Constructed Automata: {self.phi} \n"
        for i in range(len(aut.sorted_phi)):
            log_message += f"\n======== {self.phi[aut.sorted_phi[i]]} ========\n{pprint.pformat(aut.dfa[i])}\n"
        logger.debug(log_message)

        # Compute union product of DFAs
        # self._construct_semi_automaton(aut, show_progress=show_progress)
        self._construct_semi_automaton(
            aut,
            show_progress=show_progress,
            enumeration="edge-enumeration",
            backend="auto"
        )

        # Construct preference graph
        self._construct_preference_graph(aut, semantics, show_progress=show_progress)

        # Return preference automaton
        return aut

    # noinspection PyMethodMayBeStatic

    def _parse_header(self):
        """
        Parses the header of the PrefLTLf specification.

        The header should be in the form: `prefltlf <num_formula>`.

        :raises ValueError: If the header is not well-formed or if the formula type is not 'prefltlf'.
        :return: A list containing the formula type and the number of formulas.
        :rtype: list
        """
        stmts = (line.strip() for line in self.raw_spec.split("\n"))
        stmts = [line for line in stmts if line and not line.startswith("#")]
        header = stmts[0]
        header = header.split(" ")

        if len(header) != 2:
            raise ValueError(f"Ill-formatted specification file. Header should be `prefltlf <num_formula>`, "
                             f"where specifying <num_formula> is optional.")

        if header[0].strip().lower() != "prefltlf":
            raise ValueError(f"Not a PrefLTLf formula. Likely a '{header[0].strip().lower()}' formula.")

        header[1] = int(header[1].strip())
        return header

    # noinspection PyMethodMayBeStatic

    def _parse_ltlf(self, raw_spec):
        """
        Parses the raw specification into LTLf formulas and extracts atomic propositions.

        :param raw_spec: List of LTLf formula strings.
        :type raw_spec: list
        :return: A tuple containing a set of atomic propositions and a list of parsed LTLf formulas.
        :rtype: tuple(set, list)
        """
        atoms = set()
        phi = list()
        for formula in raw_spec:
            ltlf = PARSER(formula.strip())
            phi.append(ltlf)
            atoms.update(set(ltlf.find_labels()))
        return atoms, phi

    # noinspection PyMethodMayBeStatic

    def _parse_relation(self, raw_spec):
        """
        Parses the relation part of the PrefLTLf specification.

        :param raw_spec: List of preference relations (e.g., '>, >=, ~, <>').
        :type raw_spec: list
        :raises AssertionError: If an unrecognized operator is encountered.
        :return: A set of preference relations.
        :rtype: set
        """
        relation = set()
        for formula in raw_spec:
            # Split line into operator, left formula, right formula.
            rel = formula.split(",")

            # Determine operator
            pref_type = rel[0].strip()
            assert pref_type in [">", ">=", "~",
                                 "<>"], f"The formula is ill-formed. Unrecognized operator `{pref_type}`"

            # Determine formulas
            left = int(rel[1].strip())
            right = int(rel[2].strip())

            # Add relation
            relation.add((pref_type, left, right))

        return relation

    def _build_preorder(self, relation_spec):
        """
        Builds the preorder relation based on the parsed specification.

        :param relation_spec: Set of preference relations.
        :type relation_spec: set
        :return: A set representing the preorder relation.
        :rtype: set
        """
        preorder = set()

        # First, process the non-incomparability formulas because they add elements to preorder relation.
        for pref_type, phi1, phi2 in (f for f in relation_spec if f[0] != "<>"):
            assert 0 <= phi1 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

            if pref_type == ">" or pref_type == ">=":
                preorder.add((phi1, phi2))

            if pref_type == "~":
                preorder.add((phi1, phi2))
                preorder.add((phi2, phi1))

                # Second, process the incomparability formulas because it removes elements to preorder relation.
        for pref_type, phi1, phi2 in (f for f in relation_spec if f[0] == "<>"):
            assert 0 <= phi1 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_1={phi1}."
            assert 0 <= phi2 <= len(
                self.phi), f"Index of LTLf formula out of bounds. |Phi|={len(self.phi)}, phi_2={phi2}."

            if (phi1, phi2) in preorder:
                logger.warning(f"{(phi1, phi2)} is removed from preorder.")
                preorder.remove((phi1, phi2))

            if (phi2, phi1) in preorder:
                logger.warning(f"{(phi2, phi1)} is removed from preorder.")
                preorder.remove((phi2, phi1))

                # Reflexive closure
        for i in range(len(self.phi)):
            preorder.add((i, i))

            # Transitive closure
        preorder = self._transitive_closure(preorder)

        # Return preorder
        return preorder

    # noinspection PyMethodMayBeStatic

    def _transitive_closure(self, preorder):
        """
        Computes the transitive closure of a preorder relation.

        :param preorder: Set of pairs representing the preorder relation.
        :type preorder: set
        :return: A set representing the transitive closure of the preorder.
        :rtype: set
        """
        closure = set(preorder)
        while True:
            new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return closure

    def _construct_semi_automaton(
            self,
            aut,
            show_progress=False,
            use_multiprocessing=True,
            backend="auto",
            enumeration="auto"
    ):
        """
        Constructs a semi-automaton from a set of DFA states and transitions.

        :param aut: A PrefAutomaton object where the semi-automaton will be stored.
        :type aut: PrefAutomaton
        :param show_progress: Whether to display a progress bar for the construction. Default: `False`
        :type show_progress: bool
        :param backend: Whether to use spot or sympy backend for PL formula operations. Default: `auto`.
        :type show_progress: ["auto", "spot", "sympy"]
        :param use_multiprocessing: Whether to use multiprocessing for constructing edges. Default: `True`.
        :type backend: bool
        :return: None
        """
        # Options processing
        # 1. If backend is auto, determine which backend to use. Otherwise, check if selected backend is available.
        assert backend in ["auto", "spot", "sympy"], \
            (f"Unsupported backend for semi-automaton transition processing. "
             f"Expected ['auto', 'spot', 'sympy'], received '{backend}'.")

        if backend == "auto":
            if "spot" in sys.modules:
                backend = "spot"
            else:  # "sympy" in sys.modules:
                backend = "sympy"
            logger.info(f"Semi-automaton construction: {backend=}.")

        # 2. Determine use of parallelism and corresponding parameters
        if use_multiprocessing is True:
            cpu_count = os.cpu_count()
            logger.info(f"Semi-automaton construction: {use_multiprocessing=}, {cpu_count=}.")

        # 3. Determine which approach to use: alphabet-enumeration or edge-enumeration or no-enumeration?
        # Note: The bottleneck is evaluating a PL formula labeling DFA transitions.
        #   We implement a code optimization based on the following observation.
        #   The upper-bound on number of edges is the smaller quantity between
        #   a) |alphabet| * SUM_i(E_i), since from each state there is exactly one edge enabled for a given symbol.
        #   b) |E_1| * ... * |E_n|: Given a state in semi-automaton, all combinations of out-edges in component DFA.
        #   In each case, the method for evaluating validity of edge combintion is different.
        #   We will use the approach that requires less computation.
        # When user has specified enumeration = "none", we will disregard above optimization.
        if enumeration == "none":
            approach = "no-enumeration"
            logger.info(
                f"Semi-automaton construction: User enforced, {approach=}."
            )
        elif enumeration == "alphabet-enumeration":
            approach = "alphabet-enumeration"
            logger.info(
                f"Semi-automaton construction: User enforced, {approach=}."
            )
        elif enumeration == "edge-enumeration":
            if self.alphabet:
                approach = "alphabet-enumeration"
                logger.error(
                    f"Since {self.alphabet=} is given, `edge-enumeration` cannot be used.\n"
                    f"Semi-automaton construction: Defaulting to {approach=}."
                )
            else:
                approach = "edge-enumeration"
                logger.info(
                    f"Semi-automaton construction: User enforced, {approach=}."
                )
        else:  # enumeration == "auto":
            if self.alphabet:
                approach = "alphabet-enumeration"
                logger.error(
                    f"Semi-automaton construction: Since {self.alphabet=} is given, defaulting to {approach=}."
                )
            else:
                num_trans = [len(trans) for d in aut.dfa for _, trans in d["transitions"].items()]
                alphabet_enum_size = len(self.alphabet) if not self.alphabet else (2 ** len(self.atoms)) * sum(num_trans)
                edge_enum_size = math.prod(num_trans)
                approach = "alphabet-enumeration" if alphabet_enum_size <= edge_enum_size else "edge-enumeration"
                logger.info(
                    f"Semi-automaton construction: {alphabet_enum_size=}, {edge_enum_size=}, determined {approach=}."
                )

        # Define a semi-automaton as networkx graph
        semi_aut = nx.MultiDiGraph()

        # Define state set or initial state, depending on selected approach.
        init_state = tuple([dfa['init_state'] for dfa in aut.dfa])
        semi_aut.add_node(init_state)
        semi_aut.graph["init_state"] = init_state

        if approach != "no-enumeration":
            states = itertools.product(*(d["states"] for d in aut.dfa))
            semi_aut.add_nodes_from(states)

        # Invoke appropriate transition constructor
        if approach == "no-enumeration":
            semi_aut = self._construct_sa_no_enum(semi_aut, aut.dfa, backend, show_progress=False)
        elif approach == "alphabet-enumeration":
            semi_aut = self._construct_sa_alphabet_enum(
                semi_aut,
                aut.dfa,
                cpu_count,
                backend,
                show_progress=False
            )
        else:  # approach == "edge-enumeration":
            semi_aut = self._construct_sa_edge_enum(semi_aut, aut.dfa, show_progress=False)

        # Prune semi-automaton, depending on selected approach.
        if approach != "no-enumeration":
            bfs_edges = list(nx.edge_bfs(semi_aut, source=init_state))
            pruned_semi_aut = nx.edge_subgraph(semi_aut, bfs_edges)
        else:
            pruned_semi_aut = semi_aut

        # Update `aut` data structure.
        for node in pruned_semi_aut.nodes():
            aut.add_state(node)
        for u, v, cond in pruned_semi_aut.edges(keys=True):
            aut.add_transition(u, v, cond)

    def _construct_sa_no_enum(self, semi_aut, dfa, backend, show_progress=False):
        # Get initial state
        q0 = semi_aut.graph["init_state"]

        # Setup backend for PL formula processing
        eval_func = spot_eval if backend == "spot" else sympy_eval
        simplify_func = spot_simplify if backend == "spot" else sympy_simplify

        # Visit reachable states
        queue = [q0]
        explored = set()
        transitions = set()
        with (tqdm(total=len(queue), desc="Constructing semi-automaton", disable=not show_progress) as pbar):
            while queue:
                # Visit next state
                q = queue.pop()
                explored.add(q)

                # Add state to preference automaton
                if not semi_aut.has_node(q):
                    semi_aut.add_node(q)

                # If alphabet is specified, expand `q` using alphabet.
                if self.alphabet:
                    # For each symbol in alphabet
                    for true_atoms in self.alphabet:
                        # Evaluate all component DFA edges from q[i] under this symbol.
                        next_q = tuple()
                        conditions = []
                        for i in range(len(q)):
                            trans_qi = dfa[i]['transitions'][q[i]]

                            for cond, v in trans_qi.items():
                                if eval_func(cond, true_atoms, self.atoms):
                                    next_q += (v,)
                                    conditions.append(cond)
                                    break

                        # Determine (unique) next state under the symbol and add transition.
                        # cond = "(" + ") & (".join(conditions) + ")"
                        assert len(next_q) == len(q), \
                            f"Problem constructing next state. Probably, some DFA is incomplete."
                        cond = " & ".join(true_atoms) + " & ".join(f"!{p}" for p in self.atoms - true_atoms)
                        transitions.add((q, next_q, cond))
                        semi_aut.add_edge(q, next_q, key=cond)

                        # Manage queue
                        if next_q not in explored:
                            queue.append(next_q)

                else:  # not self.alphabet
                    # Construct set of all candidate transitions
                    transitions_from_qi = [dfa[i]['transitions'][q[i]] for i in range(len(dfa))]

                    # Evaluate each one for non-trivial condition.
                    for trans_bunch in itertools.product(*transitions_from_qi):
                        conditions = [trans_i["cond"] for trans_i in trans_bunch]
                        cond = "(" + ") & (".join(conditions) + ")"
                        cond = simplify_func(cond)
                        if cond is not False:
                            # Construct next state representation
                            next_q = tuple(trans_i["p"] for trans_i in trans_bunch)
                            transitions.add((q, next_q, cond))
                            semi_aut.add_edge(q, next_q, key=cond)

                            # Manage queue
                            if next_q not in explored:
                                queue.append(next_q)

                # Update tqdm progress bar
                pbar.update(1)
                pbar.total = len(queue) + pbar.n
                pbar.refresh()

        return semi_aut

    def _construct_sa_alphabet_enum(self, semi_aut, dfa, cpu_count, backend, show_progress=False):
        # Setup backend for PL formula processing
        eval_func = spot_eval if backend == "spot" else sympy_eval

        # Construct tuples for evaluation: (i, q[i], conditions[q[i]], true_atoms)
        args = []
        for i in range(len(dfa)):
            for q in dfa[i]["states"]:
                args.append((i, q, dfa[i]["transitions"][q]))
        alphabet = self.alphabet if self.alphabet else utils.powerset(self.atoms)
        args = list(itertools.product(args, alphabet))
        args = [inp + (eval_func, self.atoms) for inp in args]

        # Process all states and alphabet combinations; each element is `next_q`
        result = list(pqdm(args, _worker_alphabet_enum, n_jobs=cpu_count, disable=not show_progress))

        # Create map: {i: {q: {<true_atoms: tuple>: p}}}
        helper_map = {
            i: {
                q: {
                    tuple(true_atoms): None for true_atoms in alphabet
                }
                for q in dfa[i]["states"]
            }
            for i in range(len(dfa))
        }

        for idx in range(len(args)):
            (i, q, trans_q), true_atoms, _, _ = args[idx]
            p = result[idx]
            helper_map[i][q][tuple(sorted(true_atoms))] = p

        # Create transitions
        for q in semi_aut.nodes():
            for true_atoms in alphabet:
                p = tuple(helper_map[i][q[i]][tuple(sorted(true_atoms))] for i in range(len(dfa)))
                cond = " & ".join(true_atoms) + " & ".join(f"!{p}" for p in self.atoms - true_atoms)
                semi_aut.add_edge(q, p, key=cond)

        # # =========================================================
        # state_alphabet = list(itertools.product(semi_aut.nodes(), alphabet))
        # args = (inp + (eval_func, dfa) for inp in state_alphabet)
        # next_states = list(pqdm(args, self._worker_alphabet_enum, n_jobs=cpu_count, disable=not show_progress))
        # for i in range(len(state_alphabet)):
        #     true_atoms = state_alphabet[i][1]
        #     cond = " & ".join(true_atoms) + " & ".join(f"!{p}" for p in self.atoms - true_atoms)
        #     semi_aut.add_edge(state_alphabet[i][0], next_states[i], key=cond)

        return semi_aut

    def _construct_sa_edge_enum(self, semi_aut, dfa, show_progress=False):
        raise ValueError("Alphabet needs to be accounted for.")
        # Construct set of candidate transitions
        transitions = list()
        for state in semi_aut.nodes():
            component_trans = list()
            for i in range(len(dfa)):
                # dfa = dfa[i]
                trans = dfa[i]["transitions"][state[i]]
                component_trans.append([(state[i], v, a) for a, v in trans.items()])
            transitions += list(itertools.product(*component_trans))

        # Set up concurrent.futures multiprocessing
        cpu_count = os.cpu_count()
        transitions = pqdm(transitions, check_plformula, n_jobs=cpu_count, disable=not show_progress)

        # Construct transitions of product semi-automaton
        for trans in transitions:
            components, cond, is_valid = trans
            if not is_valid:
                continue
            src_state = tuple(components[i][0] for i in range(len(components)))
            tgt_state = tuple(components[i][1] for i in range(len(components)))
            semi_aut.add_edge(src_state, tgt_state, cond)

        return semi_aut

    def _construct_semi_automaton_2(self, aut, show_progress=False):
        """
        Constructs a semi-automaton from a set of DFA states and transitions.

        :param aut: A PrefAutomaton object where the semi-automaton will be stored.
        :type aut: PrefAutomaton
        :param show_progress: Whether to display a progress bar for the construction.
        :type show_progress: bool
        :return: None
        """
        dfa = aut.dfa

        # Initial state
        q0 = tuple([dfa['init_state'] for dfa in dfa])
        aut.init_state = aut.add_state(q0)

        # Visit reachable states
        queue = [q0]
        explored = set()
        transitions = set()
        with tqdm(total=len(queue), desc="Constructing semi-automaton", disable=not show_progress) as pbar:
            while queue:
                # Visit next state
                q = queue.pop()
                explored.add(q)

                # Add state to preference automaton
                aut.add_state(q)

                # Add transitions to product dfa
                # Pick one outgoing edge from each of the sub-DFA states in q.
                for condition_on_edges in itertools.product(
                        *[dfa[i]['transitions'][q[i]].keys() for i in range(len(dfa))]
                ):
                    # Define condition for synchronous transition of selected edges (AND-ing of all conditions)
                    cond = sympy.sympify(("(" + ") & (".join(condition_on_edges) + ")").replace("!", "~")).simplify()

                    # If alphabet is provided, require that at least one symbol enables the condition.
                    if (
                            self.alphabet and
                            all(
                                not cond.subs({atom: True if atom in true_atoms else False for atom in self.atoms})
                                for true_atoms in self.alphabet
                            )
                    ):
                        continue

                        # If label is false, then the synchronous transition is not valid.
                    if cond == sympy.false:
                        continue

                        # Otherwise, add transition
                    cond = (str(cond).replace("~", "!").
                            replace("True", "true").
                            replace("False", "false"))

                    q_next = tuple([dfa[i]['transitions'][q[i]][condition_on_edges[i]] for i in range(len(dfa))])
                    if q_next not in explored:
                        queue.append(q_next)

                        # Add transition to preference automaton
                    transitions.add((q, q_next, cond))

                # Update tqdm progress bar
                pbar.update(1)
                pbar.total = len(queue) + pbar.n
                pbar.refresh()

        for q, p, cond in transitions:
            aut.add_transition(q, p, cond)

    def _construct_semi_automaton_3(self, aut, show_progress=False):
        """
        Constructs a semi-automaton from a set of DFA states and transitions.

        :param aut: A PrefAutomaton object where the semi-automaton will be stored.
        :type aut: PrefAutomaton
        :param show_progress: Whether to display a progress bar for the construction.
        :type show_progress: bool
        :return: None
        """
        # Get all DFAs
        dfa = aut.dfa

        # Compute all states
        states = list(itertools.product(*(d["states"] for d in dfa)))
        for state in states:
            aut.add_state(state)

        # Initial state
        q0 = tuple([dfa['init_state'] for dfa in dfa])
        aut.init_state = aut.add_state(q0)

        # TODO: Set up parallel processing here for edges.
        #   1. Alphabet size
        #   2. Value of |Product of Outdegree(s_i)|, given s = (s_1, s_2, ..., s_n).
        #   If alphabet size is small, then fix a symbol and evaluate edge within individual DFA
        #   Else, enumerate (s, e_1, ..., e_n) tuples and run a parallel processing loop.
        # TODO: Provide an option to force a certain behavior.

        # Estimate alphabet size
        num_trans_in_dfas = [len(trans) for d in dfa for _, trans in d["transitions"].items()]
        alphabet_size = 2 ** len(self.atoms) if not self.alphabet else len(self.alphabet)
        alphabet_size = alphabet_size * sum(num_trans_in_dfas)
        edge_set_size = math.prod(num_trans_in_dfas)
        print(f"{alphabet_size=}, {edge_set_size=}")

        # Initialize transitions
        if alphabet_size < edge_set_size:
            # For each alphabet, for each DFA, create a state-enabled_transition mapping.
            #   Create a list of (dfa-index, transition, symbol).
            #   Evaluate in parallel which transitions are enabled under the given symbol.
            #   Construct a map: [{symbol: {state: transition}}] for each DFA.

            transitions = list()
            for i in range(len(aut.dfa)):
                dfa = aut.dfa[i]
                transitions += [(i, (u, v, a)) for u, trans in dfa["transitions"].items() for a, v in trans.items()]
            alphabet = self.alphabet if self.alphabet else utils.powerset(self.atoms)
            transitions = list(itertools.product(transitions, alphabet))

            # Set up concurrent.futures multiprocessing
            cpu_count = os.cpu_count()
            args = ((trans, self.atoms, symbol) for trans, symbol in transitions)
            transitions = pqdm(args, eval_plformula, n_jobs=cpu_count)

            # TODO. Reconstruct transitions
            pprint.pprint(transitions)

        else:
            # Construct set of candidate transitions
            transitions = list()
            for state in states:
                component_trans = list()
                for i in range(len(aut.dfa)):
                    dfa = aut.dfa[i]
                    trans = dfa["transitions"][state[i]]
                    component_trans.append([(state[i], v, a) for a, v in trans.items()])
                transitions += list(itertools.product(*component_trans))

            # Set up concurrent.futures multiprocessing
            cpu_count = os.cpu_count()
            transitions = pqdm(transitions, check_plformula, n_jobs=cpu_count)

            # Construct transitions of product semi-automaton
            for trans in transitions:
                components, cond, is_valid = trans
                if not is_valid:
                    continue
                src_state = tuple(components[i][0] for i in range(len(components)))
                tgt_state = tuple(components[i][1] for i in range(len(components)))
                aut.add_transition(src_state, tgt_state, cond)
            pprint.pprint(aut.transitions)
        exit(0)

        # Visit reachable states
        queue = [q0]
        explored = set()
        transitions = set()
        with tqdm(total=len(queue), desc="Constructing semi-automaton", disable=not show_progress) as pbar:
            while queue:
                # Visit next state
                q = queue.pop()
                explored.add(q)

                # Add state to preference automaton
                aut.add_state(q)

                # Add transitions to product dfa
                # Pick one outgoing edge from each of the sub-DFA states in q.
                for condition_on_edges in itertools.product(
                        *[dfa[i]['transitions'][q[i]].keys() for i in range(len(dfa))]
                ):
                    # Define condition for synchronous transition of selected edges (AND-ing of all conditions)
                    cond = sympy.sympify(("(" + ") & (".join(condition_on_edges) + ")").replace("!", "~")).simplify()

                    # If alphabet is provided, require that at least one symbol enables the condition.
                    if (
                            self.alphabet and
                            all(
                                not cond.subs({atom: True if atom in true_atoms else False for atom in self.atoms})
                                for true_atoms in self.alphabet
                            )
                    ):
                        continue

                    # If label is false, then the synchronous transition is not valid.
                    if cond == sympy.false:
                        continue

                    # Otherwise, add transition
                    cond = (str(cond).replace("~", "!").
                            replace("True", "true").
                            replace("False", "false"))

                    q_next = tuple([dfa[i]['transitions'][q[i]][condition_on_edges[i]] for i in range(len(dfa))])
                    if q_next not in explored:
                        queue.append(q_next)

                        # Add transition to preference automaton
                    transitions.add((q, q_next, cond))

                # Update tqdm progress bar
                pbar.update(1)
                pbar.total = len(queue) + pbar.n
                pbar.refresh()

        for q, p, cond in transitions:
            aut.add_transition(q, p, cond)

    def _construct_preference_graph(self, aut, semantics, show_progress=False):
        """
        Constructs a preference graph based on the automaton and the given semantics.

        :param aut: A PrefAutomaton object where the preference graph will be stored.
        :type aut: PrefAutomaton
        :param semantics: A semantics function determining how preferences are interpreted.
        :type semantics: function
        :param show_progress: Whether to display a progress bar for the construction.
        :type show_progress: bool
        :return: None
        """
        # dfa = self.dfa
        pg = nx.MultiDiGraph()

        # Create partition and add nodes
        for qid, data in tqdm(
                aut.get_states(data=True),
                desc="Constructing preference graph nodes",
                disable=not show_progress
        ):
            q = data["name"]
            outcomes = utils.outcomes(aut.sorted_phi, aut.dfa, q)
            if semantics in PrefLTLf.MAXIMAL_SEMANTICS:
                outcomes = utils.maximal_outcomes(self.relation, outcomes)

            cls = utils.vectorize(aut.sorted_phi, aut.dfa, outcomes)
            # cls_id = aut.add_class(cls)

            # Add node to temporary graph
            if not pg.has_node(cls):
                pg.add_node(cls, partition={qid})
            else:
                pg.nodes[cls]["partition"].add(qid)

            # aut.add_state_to_class(cls_id, q)

        # Create edges
        # for source_id, target_id in itertools.product(aut.pref_graph.nodes(), aut.pref_graph.nodes()):
        for source, target in tqdm(
                itertools.product(pg.nodes(), pg.nodes()),
                total=pg.number_of_nodes() ** 2,
                desc="Constructing preference graph edges",
                disable=not show_progress
        ):
            # source = aut.get_class_name(source_id)
            # target = aut.get_class_name(target_id)
            if semantics(aut.sorted_phi, self.relation, source, target):
                # aut.add_pref_edge(source_id, target_id)
                pg.add_edge(source, target)

        # Merge partitions if two nodes are indifferent under constructed edge relation
        scc = nx.strongly_connected_components(pg)

        phi = [self.phi[ltlf_id] for ltlf_id in sorted(self.phi.keys())]
        state2node = dict()
        for component in scc:
            # Define a class name and class id
            cls_name = []
            for cls in component:
                cls_name.append(
                    " & ".join([str(phi[i]) for i in range(len(phi)) if cls[i] == 1])
                )

            # Determine class name and add class to aut
            cls_name = " | ".join((f"({x})" for x in cls_name))
            cls_name = str(PARSER(cls_name))
            cls_id = aut.add_class(cls_name)

            # Add states to partition
            for cls in component:
                for q in pg.nodes[cls]["partition"]:
                    aut.add_state_to_class(cls_id, q)
                    state2node[cls] = cls_id

        for u, v in pg.edges():
            aut.add_pref_edge(state2node[u], state2node[v])

    def _construct_spec_ir(self):
        """
        Constructs an intermediate representation (IR) of the PrefLTLf specification.

        The IR consists of:
            1. LTLf formulas parsed from the specification.
            2. The preference relation between formulas.

        :raises ValueError: If a formula cannot be parsed.
        :return: A tuple containing a dictionary of LTLf formulas and a list of preference relations.
        :rtype: tuple(dict, list)
        """
        # Initialize outputs
        phi = dict()

        # Extract non-header lines from spec
        stmts = (line.strip() for line in self.raw_spec.split("\n"))
        stmts = [line for line in stmts if line and not line.startswith("#")]
        stmts = stmts[1:]

        # Extract header
        header = self._parse_header()
        n_phi = header[1]

        # Construct phi: the set of LTLf formulas.
        for i in range(n_phi):
            try:
                phi[i] = PARSER(stmts[i].strip())
            except Exception as err:
                if isinstance(err, lark.exceptions.UnexpectedCharacters):
                    # logger.debug(str(err))
                    raise ValueError(
                        f"Could not parse '{stmts[i].strip()}'. Possible reasons:\n"
                        f"1. Did you supply {n_phi} formulas as suggested by the specification header?\n"
                        f"2. LTLf formula is ill-formed"
                    )

        # Eliminate formulas with equivalent languages
        equiv = self._find_equivalent_ltlf(phi)
        compact_phi, replacement = self._process_equiv_ltlf(phi, equiv)

        # Collect atomic preferences
        spec_ir = set()
        for atomic_pref in stmts[n_phi:]:
            pref_type, l_index, r_index = atomic_pref.split(",")
            l_index = int(l_index.strip())
            r_index = int(r_index.strip())
            assert pref_type.strip() in [">", ">=", "~", "<>"], \
                f'{pref_type.strip()} is not a valid. Valid preference operators are {[">", ">=", "~", "<>"]}.'
            assert l_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, l_index={l_index}."
            assert r_index in phi, f"Index of LTLf formula out of bounds. |Phi|={len(phi)}, r_index={l_index}."

            if l_index in replacement:
                l_index = replacement[l_index]
            if r_index in replacement:
                r_index = replacement[r_index]

            spec_ir.add((pref_type, l_index, r_index))

        # Log constructed representation
        log_string = [
            f"{l_index}:{compact_phi[l_index]}  {pref_type}  {r_index}:{compact_phi[r_index]}"
            for pref_type, l_index, r_index in spec_ir
        ]
        logger.debug(f"Intermediate representation based on raw input: \n{pprint.pformat(log_string)}")

        # Return intermediate representation
        return compact_phi, list(spec_ir)

    # noinspection PyMethodMayBeStatic
    def _auto_complete(self, phi, spec_ir, auto_complete):
        """
        Handles the automatic completion of a set of LTLf formulas `phi`.

        :param phi: Dictionary of LTLf formulas.
        :param spec_ir: Intermediate representation of preferences.
        :param auto_complete: Specifies the auto-completion method to apply ("minimal" or "incomparable").
        :return: Tuple containing the updated `phi` and `spec_ir`.
        """
        # Determine completion formula
        completion_formula = " | ".join([f"({varphi})" for varphi in phi.values()])
        completion_formula = f"!({completion_formula})"
        completion_formula = PARSER(completion_formula)

        # Add completion formula to phi.
        new_id = max(phi.keys()) + 1
        phi.update({new_id: completion_formula})

        # Update spec_ir based on input auto-complete method.
        #   In case auto-complete type is incomparable, no update to spec_ir is necessary.
        if auto_complete == "minimal":
            for varphi in phi.keys():
                spec_ir.append((">=", varphi, new_id))

        # Log constructed representation
        log_string = [
            f"{l_index}:{phi[l_index]}  {pref_type}  {r_index}:{phi[r_index]}"
            for pref_type, l_index, r_index in spec_ir
        ]
        logger.debug(f"Intermediate representation based on raw input: {log_string}")

        # Return intermediate representation
        return phi, spec_ir

    # noinspection PyMethodMayBeStatic
    def _is_lang_complete(self, phi):
        """
        Checks if the language of a set of LTLf formulas is complete, i.e., whether it covers all possible behaviors.

        :param phi: Dictionary of LTLf formulas.
        :return: Boolean indicating whether the language is complete.
        """
        formula = " | ".join([f"({varphi})" for varphi in phi.values()])
        equiv_formula = PARSER(formula)
        dfa = utils.ltlf2dfa(equiv_formula)

        # Check for universally true DFA
        if len(dfa["states"]) == 1 and len(dfa["final_states"]) == 1:
            return True
        return False

    def _build_partial_order(self, phi, pref_stmts):
        """
        Constructs a partial order based on the provided preference statements and formulas.

        :param phi: Dictionary of LTLf formulas.
        :param pref_stmts: List of preference statements in the form (op, l_index, r_index).
        :return: Tuple containing the updated `phi` and the constructed partial order.
        """
        # Construct P, P', I, J sets
        set_w = set()
        set_p = set()
        set_i = set()
        set_j = set()
        for pref_type, varphi1, varphi2 in pref_stmts:
            if pref_type == ">":
                set_p.add((varphi1, varphi2))

            elif pref_type == ">=":
                set_w.add((varphi1, varphi2))

            elif pref_type == "~":
                set_i.add((varphi1, varphi2))
                set_i.add((varphi2, varphi1))

            elif pref_type == "<>":
                set_j.add((varphi1, varphi2))
                set_j.add((varphi2, varphi1))

        logger.debug(f"Clauses from intermediate representation:\n{set_w=} \n{set_p=} \n{set_i=} \n{set_j=}")

        # Resolve W into P, I, J
        for varphi1, varphi2 in set_w:
            if (varphi1, varphi2) in set_j:
                raise ValueError(
                    f"Inconsistent specification: "
                    f"{phi[varphi1]} >= {phi[varphi2]} and {phi[varphi1]} <> {phi[varphi2]}."
                )

            elif (varphi2, varphi1) in set_p:
                raise ValueError(
                    f"Inconsistent specification: "
                    f"{phi[varphi1]} >= {phi[varphi2]} and {phi[varphi2]} > {phi[varphi1]}."
                )

            elif (varphi1, varphi2) in set_p | set_i:
                pass

            elif (varphi2, varphi1) in set_w:
                set_i.add((varphi1, varphi2))

            else:
                set_p.add((varphi1, varphi2))

        # Transitive closure
        set_p = self._transitive_closure(set_p)
        set_i |= {(varphi, varphi) for varphi in phi}
        set_i = self._transitive_closure(set_i)
        logger.debug(f"Clauses after applying transitive and reflexive closures:\n{set_p=} \n{set_i=} \n{set_j=}")

        # Consistency check
        if any((varphi1, varphi1) in set_p for varphi1 in phi):
            graph_ = nx.DiGraph()
            graph_.add_nodes_from(phi.keys())
            graph_.add_edges_from([(j, i) for i, j in set_p])
            cycles = sorted(nx.simple_cycles(graph_))
            cycles = [list(map(phi.get, cycle)) for cycle in cycles if len(cycle) >= 3]
            raise ValueError(
                f"Inconsistent specification: Cyclic preferences detected in P. "
                f"Cycles after transitive closure: \n{pprint.pformat(cycles)}"
            )

        if any((varphi2, varphi1) in set_p for varphi1, varphi2 in set_p):
            raise ValueError(f"Inconsistent specification: Strictness violated. "
                             f"{[[(p1, p2), (p2, p1)] for p1, p2 in set_p if (p2, p1) in set_p]} in P.")

        if set.intersection(set_p, set_i):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_i)} common in P and I.")

        if set.intersection(set_p, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_p, set_j)} common in P and J.")

        if set.intersection(set_i, set_j):
            raise ValueError(f"Inconsistent specification: {set.intersection(set_i, set_j)} common in I and J.")

        # # Merge any specifications with same language.
        # equiv = self._find_equivalent_ltlf(phi)
        # phi, model = self._process_equiv_ltlf(phi, set.union(set_p, set_i), equiv)
        # logger.debug(f"Merge language equivalent formulas:\n{phi=} \n{model=}")

        # Apply reflexive closure
        model = set.union(set_p, set_i)
        model.update({(i, i) for i in phi.keys()})

        # Apply transitive closure
        # model = util.transitive_closure(model)
        # print(phi)
        # print(model)

        # If spec is a preorder, but not a partial order, then construct a partial order. [add option to skip this]
        phi, model = self._pre_to_partial_order(phi, model)

        log_model = {f"({phi[l_index]}, {phi[r_index]})" for l_index, r_index in model}
        logger.debug(f"Partial order for input specification:\n{phi=}\n{model=}\nmodel={log_model}")
        return phi, model

    # noinspection PyMethodMayBeStatic
    def _find_equivalent_ltlf(self, phi):
        """
        Identifies equivalent LTLf formulas from the given set of formulas.

        :param phi: Dictionary of LTLf formulas.
        :return: A set of tuples containing pairs of equivalent formulas.

        .. note:: O(n^2) complexity.
        """
        # Initialize output
        equiv_formulas = set()

        #  Compare every formula with every other formula to identify equivalent LTLf formulas.
        indices = list(phi.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Approach: Construct equivalence LTLf formula.
                #   Check if DFA is universally true, i.e., has a single state which is accepting.

                # Equivalence formula
                equiv_formula = PARSER(f"{phi[indices[i]]} <-> {phi[indices[j]]}")
                dfa = utils.ltlf2dfa(equiv_formula)

                # Check for universally true DFA
                if len(dfa["states"]) == 1 and len(dfa["final_states"]) == 1:
                    equiv_formulas.add((indices[i], indices[j]))

                # print(equiv_formula, dfa)

        return equiv_formulas

    # noinspection PyMethodMayBeStatic
    def _process_equiv_ltlf(self, phi, equiv):
        """
        Processes and merges equivalent LTLf formulas into a single representative formula.

        :param phi: Dictionary of LTLf formulas.
        :param equiv: Set of tuples representing equivalent formulas.
        :return: Tuple containing the new set of formulas and a dictionary for formula replacements.
        """
        # Construct equivalence graph
        equiv_graph = nx.DiGraph()

        for u, v in equiv:
            equiv_graph.add_edge(u, v)
            equiv_graph.add_edge(v, u)

        # equiv_graph.add_edges_from(equiv)
        scc = list(nx.strongly_connected_components(equiv_graph))

        # Construct new phi
        idx_to_remove = set()
        replacement = dict()
        for component in scc:
            keep_f = min(component)
            for f in component - {keep_f}:
                replacement[f] = keep_f
            idx_to_remove.update(component - {min(component)})

        n_phi = {k: v for k, v in phi.items() if k not in idx_to_remove}

        return n_phi, replacement

    # noinspection PyMethodMayBeStatic
    def _pre_to_partial_order(self, phi, preorder):
        """
        Converts a preorder (which may contain indifferent alternatives) to a partial order by resolving indifference.

        :param phi: Dictionary of LTLf formulas.
        :param preorder: Set representing the preorder relations.
        :return: Tuple containing the updated `phi` and the constructed partial order.
        """
        # Initialize output
        partial_order = set()

        # Construct preorder graph (might contain cycles)
        order_graph = nx.DiGraph()
        order_graph.add_edges_from(preorder)

        # Find strongly connected components
        scc = nx.strongly_connected_components(order_graph)

        # Replace indifferent formulas with their disjunction
        max_index = max(phi.keys())
        replace = dict()

        for component in scc:
            if len(component) >= 2:
                # Construct disjunction of indifferent formulas in the `component`
                formulas = (f"({phi[i]})" for i in component)
                replacement_ltlf = PARSER(" | ".join(formulas))

                # Add new formula to phi
                phi[max_index + 1] = replacement_ltlf

                # Remove indifferent formulas from phi
                max_index = max_index + 1
                for i in component:
                    phi.pop(i)
                    replace[i] = max_index

        # Construct partial order from preorder by replacing indifferent formulas
        for i, j in preorder:
            if i in replace:
                i = replace[i]
            if j in replace:
                j = replace[j]
            partial_order.add((i, j))

        # Return
        return phi, partial_order

    def _construct_sa_edges_given_alphabet(self, aut, states, show_progress=False):
        # Construct alphabet x states
        alphabet = self.alphabet if self.alphabet else utils.powerset(self.atoms)
        alphabet_states = list(itertools.product(alphabet, states))

        # TODO. deserialization has problem! Most likely mona is inaccessible when deserializing.
        # TODO. Investigate why the speed is so slow.
        #   Try spot instead of sympy. Might be faster. 

        # Set up concurrent.futures multiprocessing
        # cpu_count = os.cpu_count()
        # with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        #     transitions = set(executor.map(self._helper, alphabet_states))
        transitions = self._helper(alphabet_states, aut)

    def _helper(self, pairs, aut):
        transitions = set()
        for symbol, state in tqdm(pairs):
            # Construct substitution map
            subs_map = {atom: True if atom in symbol else False for atom in self.atoms}

            # Get edge corresponding to component dfa states that evaluates to true under alphabet
            n_state = tuple()
            condition_on_edges = []
            for i in range(len(state)):
                trans_i = aut.dfa[i]["transitions"][state[i]]
                for cond, n_state_i in trans_i.items():
                    cond = sympy.sympify(cond.replace("!", "~"))  # .simplify()
                    if cond.subs(subs_map) == sympy.true:
                        n_state += (n_state_i,)
                        condition_on_edges.append(str(cond))
                        break

            # Ensure next state is appropriately constructed
            assert len(n_state) == len(state), "Problem during next state construction."

            # Construct condition on transition
            cond = sympy.sympify(("(" + ") & (".join(condition_on_edges) + ")").replace("!", "~")).simplify()
            cond = (str(cond).replace("~", "!").
                    replace("True", "true").
                    replace("False", "false"))
            # print(cond)
            transitions.add((state, n_state, cond))

        return transitions


class PrefAutomaton:
    def __init__(self):
        # Automaton structure
        self.states = dict()
        self.atoms = set()
        self.alphabet = set()
        self.transitions = dict()
        self.init_state = None
        self.pref_graph = nx.MultiDiGraph()
        self.phi = dict()  # Map {formula-id: formula-str}
        self.sorted_phi = list()  # Ordered list of formulas
        self.dfa = list()  # Ordered list of DFAs corresponding to sorted_formulas

        # Helper attributes
        self._num_states = 0
        self._num_nodes = 0
        self._inv_state = dict()
        self._inv_nodes = dict()

    def __eq__(self, other):
        if isinstance(other, PrefAutomaton):
            return (
                    self.states == other.states and
                    self.atoms == other.atoms and
                    self.alphabet == other.alphabet and
                    self.transitions == other.transitions and
                    self.init_state == other.init_state and
                    self.pref_graph.nodes(data=True) == other.pref_graph.nodes(data=True) and
                    set(self.pref_graph.edges()) == set(other.pref_graph.edges())
            )
        return False

    def __str__(self):
        return pprint.pformat(self.serialize())

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        self.states = {int(k): data["name"] for k, data in obj_dict["states"].items()}
        self.atoms = set(obj_dict["atoms"])
        self.alphabet = obj_dict["alphabet"]
        self.transitions = {int(k): v for k, v in obj_dict["transitions"].items()}
        self.init_state = obj_dict["init_state"]
        self.pref_graph = nx.MultiDiGraph()

        # Construct preference graph
        for node, data in obj_dict["pref_graph"]["nodes"].items():
            self.pref_graph.add_node(int(node), **data)

        for u, vs in obj_dict["pref_graph"]["edges"].items():
            for v in vs:
                self.pref_graph.add_edge(int(u), int(v))

        self._num_states = len(self.states)
        self._num_nodes = self.pref_graph.number_of_nodes()
        self._inv_state = {tuple(v): k for k, v in self.states.items()}

    def serialize(self):
        # Construct state serialization
        state2node = {v: k for k, v in self.states.items()}
        state_dict = {k: {"name": v, "partition": None} for k, v in self.states.items()}
        for partition_label, data in self.pref_graph.nodes(data=True):
            for state in data["partition"]:
                # sid = state2node[state]
                # state_dict[sid]["partition"] = partition_label
                state_dict[state]["partition"] = partition_label

        #  Collect edges of preference graph
        pref_nodes = set(self.pref_graph.nodes())
        pref_edges = {u: set() for u in pref_nodes}
        for u, v in self.pref_graph.edges():
            pref_edges[u].add(v)

        obj_dict = {
            "states": state_dict,
            # "atoms": list(self.atoms),
            "atoms": self.atoms,
            # "alphabet": [list(symbol) for symbol in self.alphabet] if self.alphabet is not None else None,
            "alphabet": self.alphabet,
            "transitions": self.transitions,
            "init_state": self.init_state,
            "pref_graph": {
                # "nodes": {u: {"name": data["name"], "partition": {state2node[u] for u in data["partition"]}}
                "nodes": {u: {"name": data["name"], "partition": data["partition"]}
                          for u, data in self.pref_graph.nodes(data=True)},
                # "edges": {u: v for u, v in self.pref_graph.edges()}
                "edges": {u: list(v) for u, v in pref_edges.items()}
            }
        }
        return obj_dict

    @classmethod
    def deserialize(cls, obj_dict):
        aut = cls()
        aut.__setstate__(obj_dict)

        # aut.states = {int(k): v["name"] for k, v in obj_dict["states"].items()}
        # aut.atoms = set(obj_dict["atoms"])
        # aut.alphabet = set(map(tuple, obj_dict["alphabet"]))
        # aut.transitions = obj_dict["transitions"]
        # aut.init_state = obj_dict["init_state"]
        #
        # for node, data in obj_dict["pref_graph"]["nodes"].items():
        #     aut.pref_graph.add_node(int(node), **data)
        #
        # for u, vs in obj_dict["pref_graph"]["edges"].items():
        #     for v in vs:
        #         aut.pref_graph.add_edge(int(u), int(v))
        #
        # aut._num_states = len(aut.states)
        # aut._num_nodes = aut.pref_graph.number_of_nodes()
        # aut._inv_state = {tuple(v): k for k, v in aut.states.items()}
        # # aut._inv_nodes = {data["name"]: k for k, data in aut.pref_graph.nodes(data=True)}

        return aut

    def add_state(self, name):
        uid = self._inv_state.get(name, None)
        if uid is None:
            uid = self._num_states
            self.states[uid] = name
            self._inv_state[name] = uid
            self.transitions[uid] = dict()
            self._num_states += 1
        return uid

    def get_states(self, data=False):
        if data:
            state2node = {v: k for k, v in self.states.items()}
            state_dict = {k: {"name": v, "partition": None} for k, v in self.states.items()}
            for partition_label, data in self.pref_graph.nodes(data=True):
                for state in data["partition"]:
                    # sid = state2node[state]
                    state_dict[state]["partition"] = partition_label

            return state_dict.items()

        return list(self.states.keys())

    def add_transition(self, u, v, cond):
        uid = self._inv_state[u]
        vid = self._inv_state[v]
        self.transitions[uid].update({cond: vid})

    def get_state_id(self, name):
        return self._inv_state[name]

    def add_class(self, cls_name):
        cls_id = self._inv_nodes.get(cls_name, None)
        if cls_id is None:
            cls_id = self._num_nodes
            self.pref_graph.add_node(cls_id, name=cls_name, partition=set())
            self._inv_nodes[cls_name] = cls_id
            self._num_nodes += 1
        return cls_id

    def add_state_to_class(self, cls_id, q):
        self.pref_graph.nodes[cls_id]["partition"].add(q)

    def get_class_name(self, cls_id):
        return self.pref_graph.nodes[cls_id]["name"]

    def add_pref_edge(self, source_id, target_id):
        assert self.pref_graph.has_node(
            source_id), f"Cannot add preference edge. {source_id=} not in preference graph. "
        assert self.pref_graph.has_node(
            target_id), f"Cannot add preference edge. {target_id=} not in preference graph. "
        self.pref_graph.add_edge(source_id, target_id)


def eval_plformula(args):
    transition, atoms, true_atoms = args
    cond = transition[-1][-1]
    subs_map = {
        atom: True if atom in true_atoms else False
        for atom in atoms
    }

    if "spot" in sys.modules:
        # Define a transform to apply to AST of spot.formula.
        def transform(node: spot.formula):
            if node.is_literal():
                if "!" not in node.to_str():
                    if node.to_str() in subs_map.keys():
                        return spot.formula.tt()
                    else:
                        return spot.formula.ff()

            return node.map(transform)

        # Apply the transform and return the result.
        # Since every literal is replaced by true or false,
        #   the transformed formula is guaranteed to be either true or false.
        cond = spot.formula(cond)
        truth_value = True if transform(cond).is_tt() else False

    else:
        cond = sympy.sympify(cond.replace("!", "~"))  # .simplify()
        if cond.subs(subs_map) == sympy.true:
            truth_value = True
        else:
            truth_value = False

    return (transition, atoms, true_atoms), truth_value


def check_plformula(transition):
    cond = [component_trans[-1] for component_trans in transition]
    # cond = [c.replace("~", "!").replace("True", "true").replace("False", "false") for c in cond]
    cond = "(" + ") & (".join(cond) + ")"

    if "spot" in sys.modules:
        cond = spot.formula(cond).simplify()
        is_valid = True if not cond.is_ff() else False

    else:
        cond = sympy.logic.boolalg.simplify_logic(cond)
        is_valid = sympy.logic.boolalg.simplify_logic(cond) != sympy.false

    return transition, str(cond), is_valid


def spot_eval(cond, true_atoms, atoms):
    # Define a transform to apply to AST of spot.formula.
    def transform(node: spot.formula):
        if node.is_literal():
            if "!" not in node.to_str():
                if node.to_str() in true_atoms:
                    return spot.formula.tt()
                else:
                    return spot.formula.ff()

        return node.map(transform)

    f = spot.formula(cond)
    return spot.are_equivalent(transform(f), spot.formula_tt())


def sympy_eval(cond, true_atoms, atoms):
    subs_map = {
        atom: True if atom in true_atoms else False
        for atom in atoms
    }
    cond = sympy.sympify(cond.replace("!", "~"))
    return cond.subs(subs_map) == sympy.true


def spot_simplify(cond):
    return spot.formula(cond).simplify()


def sympy_simplify(cond):
    return sympy.logic.boolalg.simplify_logic(cond.replace("!", "~"))


def _worker_alphabet_enum(args):
    (i, q, trans_dict), true_atoms, eval_func, atoms = args
    for cond, p in trans_dict.items():
        if eval_func(cond, true_atoms, atoms):
            return p