# prefltlf2pdfa

Tool to translate PrefLTLf formulas to preference deterministic finite automaton (PDFA). 
A PrefLTLf formula can express a preference over LTLf formulas. 
For example, a `F(a) > F(b)` is a valid PrefLTLf formula stating satisfying 
`F(a)` is strictly preferred to satisfying `F(b)`


The tool is available at https://akulkarni.me/prefltlf2pdfa.html to try online.


_Related Publications:_ 

1. Rahmani, Hazhar, Abhishek N. Kulkarni, and Jie Fu. "Preference-Based Planning in Stochastic Environments: From Partially-Ordered Temporal Goals to Most Preferred Policies." arXiv preprint arXiv:2403.18212 (2024).
2. Kulkarni, Abhishek N., and Jie Fu. "Opportunistic Qualitative Planning in Stochastic Systems with Incomplete Preferences over Reachability Objectives." 2023 American Control Conference (ACC). IEEE, 2023.
3. Rahmani, Hazhar, Abhishek N. Kulkarni, and Jie Fu. "Probabilistic planning with partially ordered preferences over temporal goals." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.


## PrefLTLf specification

A typical structure of PrefLTLf input is as follows: 
``` 
prefltlf n
<LTLf formula>
<LTLf formula>
... 
<Type>, <int>, <int>
<Type>, <int>, <int>
...
```

PrefLTLf specification is defined in three parts:
1. *header:* The first line of the file includes the keyword `prefltlf` and specifies the number of LTLf formulas in the file.
2. *LTLf formulas:* The next `n` lines define `n` LTLf formulas. LTLf formulas are indexed from `0` to `n-1`. 
   The formula on `i`-th line of the file has index `i-1`.  
3. *Preference relation:* The remaining lines define preference relation. 
   Each preference relation is a triple of type, and indices of two LTLf formulas. 
   For example, a statement `>, i, j` means that `i`-th LTLf formula is strictly preferred to `j`-th LTLf formula. 


Notes:
* `<Type>` is either `>, >=, ~` or `<>` representing strict, weak, indifference and incomparable formulas, respectively.
* `<LTLf formula>` is a valid LTLf formula parsable by `ltlf2dfa` tool.


> Note: The parser ignores any empty lines and comments (i.e., lines starting with "#").


## Example 

As example, consider the following specification.  
(See `examples/demo.py`)

```
# Header 
prefltlf 4

# Four LTLf formulas
F a
F b
F c
!(F a | F b | F c)

# Preference relation
>, 0, 1
>=, 0, 2
>, 0, 3
~, 1, 3
>, 2, 3
```


A `PrefLTLf` formula is defined using the class `prefltlf.PrefLTLf`

```python
formula = prefltlf.PrefLTLf(spec)
```

By default, it is assumed that any symbol in `powerset({a, b, c})` is valid. 
However, in several applications such as a robot in gridworld, the set of valid symbols is small.
Thus, a PDFA accepting valid symbols may be defined by passing a `alphabet` parameter to the constructor.

```python
alphabet = [set(), {"a"}, {"b"}, {"c"}]
formula = prefltlf.PrefLTLf(spec, alphabet=alphabet)
```

A preference automaton from a formula is constructed using `translate` function. 
The `translate` function accepts a semantics as parameter.

```python
pdfa = formula.translate(semantics.semantics_mp_exists_forall)
```

The generated PDFA has the structure `(states, atoms, alphabet, transitions, init_state, pref_graph)`.
See reference [1] for details.


### Note on Inconsitent Formulas

The tool implements inconsistency checking for input specification. 
Thus, the following specification will generate an error because 
`F(a)` and `true U a` are equivalent formulas and a formula cannot be strictly preferred to itself!   


```
prefltlf 3
F a
true U a
F b
>, 0, 1
~, 0, 2
```


Similarly, inconsistency checker will complain if
* Any pair of formulas are simultaneously labeled as strict/weak preferred as well as incomparable. 
* A pair of formula is weakly preferred and the reverse is strictly preferred. 