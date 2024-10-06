"""
This module defines semantic evaluation functions for comparing two
subsets of LTLf formulas given a preorder on LTLf formulas.

Functions:
    - `semantics_forall_exists`: Evaluates under "forall_exists" condition.
    - `semantics_exists_forall`: Evaluates under "exists_forall" condition.
    - `semantics_forall_forall`: Evaluates under "forall_forall" condition.
    - `semantics_mp_forall_exists`: Evaluates under "forall_exists" condition on maximal elements of two subsets of LTLf formulas.
    - `semantics_mp_exists_forall`: Evaluates under "exists_forall" condition on maximal elements of two subsets of LTLf formulas.
    - `semantics_mp_forall_forall`: Evaluates under "forall_forall" condition on maximal elements of two subsets of LTLf formulas.
"""

import prefltlf2pdfa.utils as utils


def semantics_forall_exists(preorder, source, target):
    """
    Check if for all formulas in the source set, there exists a formula
    in the target set that satisfies the given preorder.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.
    """
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target == set():
        return True

    if sat_target == set():
        return False

    for alpha_to in sat_target:
        if len(sat_source) != 0 and not any((alpha_to, alpha_from) in preorder for alpha_from in sat_source):
            # if len(sat_source) != 0 and not any((alpha_to, alpha_from) in preorder for alpha_from in sat_source - {alpha_to}):
            return False

    return True


def semantics_exists_forall(preorder, source, target):
    """
    Check if there exists a formula in the source set such that all formulas
    in the target set satisfy the given preorder.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.
    """
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target == set():
        return True

    if sat_target == set():
        return False

    for alpha_from in sat_source:
        if not any((alpha_to, alpha_from) in preorder for alpha_to in sat_target):
            return False

    return True


def semantics_forall_forall(preorder, source, target):
    """
    Check if all formulas in the source set satisfy the preorder for all
    formulas in the target set.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.
    """
    sat_source = {i for i in range(len(source)) if source[i] == 1}
    sat_target = {i for i in range(len(target)) if target[i] == 1}

    # Force empty set to be indifferent to each other. Required for preference graph to be preorder.
    if sat_source == sat_target:
        return True

    if sat_target == set():
        return False

    for alpha_to in sat_target:
        if not all((alpha_to, alpha_from) in preorder for alpha_from in sat_source):
            return False

    return True


def semantics_mp_forall_exists(preorder, source, target):
    """
    Check if for all maximal outcomes in the source set, there exists a maximal outcome
    in the target set that satisfies the given preorder.

    This is a maximal-preorder variant of the `semantics_forall_exists` function,
    where the maximal outcomes of both the source and target sets are computed
    before evaluating the preorder condition.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.

    See Also:
        semantics_forall_exists: The base function for evaluating the condition.
        utils.maximal_outcomes: A utility function to compute maximal outcomes.
    """
    # PATCH: the input to this function is assumed to contain only maximal elements.
    #   (see _construct_preference_graph function in prefltlf.py)
    return semantics_forall_exists(preorder, source, target)


def semantics_mp_exists_forall(preorder, source, target):
    """
    Check if there exists a maximal outcome in the source set such that all maximal
    outcomes in the target set satisfy the given preorder.

    This is a maximal-preorder variant of the `semantics_exists_forall` function,
    where the maximal outcomes of both the source and target sets are computed
    before evaluating the preorder condition.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.

    See Also:
        semantics_exists_forall: The base function for evaluating the condition.
        utils.maximal_outcomes: A utility function to compute maximal outcomes.
    """
    # PATCH: the input to this function is assumed to contain only maximal elements.
    #   (see _construct_preference_graph function in prefltlf.py)
    return semantics_exists_forall(preorder, source, target)


def semantics_mp_forall_forall(preorder, source, target):
    """
    Check if all maximal outcomes in the source set satisfy the preorder
    for all maximal outcomes in the target set.

    This is a maximal-preorder variant of the `semantics_forall_forall` function,
    where the maximal outcomes of both the source and target sets are computed
    before evaluating the preorder condition.

    Args:
        preorder (list of tuple): The preorder defining the preference relation.
        source (list): List of binary values representing the source formulas.
        target (list): List of binary values representing the target formulas.

    Returns:
        bool: True if the semantic condition is satisfied, False otherwise.

    See Also:
        semantics_forall_forall: The base function for evaluating the condition.
        utils.maximal_outcomes: A utility function to compute maximal outcomes.
    """
    # PATCH: the input to this function is assumed to contain only maximal elements.
    #   (see _construct_preference_graph function in prefltlf.py)
    return semantics_forall_forall(preorder, source, target)
