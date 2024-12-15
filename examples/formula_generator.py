"""
The purpose of this script is to generate random LTL formulas for creating test cases.
We use spot's random formula generator to generate random LTL formulas, which we will treat as LTLf formulas.
"""

import spot

if __name__ == '__main__':
    num_atoms = 3
    num_formulas = 10

    # Generate random LTL formulas
    for formula in spot.randltl(num_atoms, num_formulas, seed=10).simplify().unabbreviate("WMR"):
        print(formula)

    # Generate random partial order
    import random
    import numpy as np


    def generate_random_partial_order(n):
        # Start with the identity matrix to ensure reflexivity
        partial_order = np.eye(n, dtype=bool)

        # Randomly fill the upper triangle of the matrix
        for i in range(n):
            for j in range(i + 1, n):
                if random.choice([True, False]):
                    partial_order[i, j] = True

        # Enforce transitivity: if a -> b and b -> c, then a -> c
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    partial_order[i, j] = partial_order[i, j] or (partial_order[i, k] and partial_order[k, j])

        return partial_order


    def print_partial_order(partial_order):
        n = len(partial_order)
        print("Partial Order Matrix:")
        for i in range(n):
            print("".join("1" if partial_order[i, j] else "0" for j in range(n)))

        print("\nRelations:")
        for i in range(n):
            for j in range(n):
                if partial_order[i, j]:
                    print(f">=, {i}, {j}")


    # Example usage
    partial_order = generate_random_partial_order(num_formulas)
    print_partial_order(partial_order)


