def semantics_forall_exists(preorder, source, target):
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
    return semantics_forall_exists(preorder, source, target)


def semantics_mp_exists_forall(preorder, source, target):
    return semantics_exists_forall(preorder, source, target)


def semantics_mp_forall_forall(preorder, source, target):
    return semantics_forall_forall(preorder, source, target)
