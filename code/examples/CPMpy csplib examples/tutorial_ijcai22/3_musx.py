# %%
import numpy as np
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools

from IPython.display import display, HTML # display some titles

# %%
"""
# Explaining Unsatisfiability

Errors can occur when modeling of constrained problems either an error was introduced by the modeler or the model is overconstrained. 
Debugging these models requires extracting the unsatisfiable constraints. 
In that case, a minimal subset of unsatisfiable constraints (MUS, a.k.a Minimal Unsatisfiable Subset) forms an explanation of unsatisfiability [1, 2]. 

A ___Smallest Minimal Unsatisfiable Subset (SMUS)___ forms the ___simplest/easiest___ explanation of UNSAT when all cosntraints are considered equally important.

In this notebook, we consider a deletion-based MUS extraction algorithm to illustrate the effect of assumptions, incrementality.

We apply the MUS-extraction techniques on a toy an int-variable CP model.

[1] Liffiton, M. H., & Sakallah, K. A. (2008). Algorithms for computing minimal unsatisfiable subsets of constraints. Journal of Automated Reasoning, 40(1), 1-33.

[2] Reiter, R. (1987). A theory of diagnosis from first principles. Artificial intelligence, 32(1), 57-95.

[3] Ignatiev, A., et al. "Smallest MUS extraction with minimal hitting set dualization." International Conference on Principles and Practice of Constraint Programming. Springer, Cham, 2015.

"""

# %%
"""
## Unsatisfiable int model
"""

# %%
def unsat_int_model():

    x = intvar(-9, 9, name="x")
    y = intvar(-9, 9, name="y")

    m = Model(
        x < 0, 
        x < 1,
        x > 2,
        x == 4,
        y == 4, 
        (x + y > 0) | (y < 0),
        (y >= 0) | (x >= 0),
        (y < 0) | (x < 0),
        (y > 0) | (x < 0),
        AllDifferent(x,y) # invalid for musx_assum
    )
    assert (m.solve() is False)

    return m

# %%
"""
## Deletion-based mus extraction algorithm

The MUSX algorithm repeatedly selects one constraint to remove from the problem constraints. 

- If the problem is ___SAT___ when removing the constraint, then the constraint should remain in the problem.

- However if the problem is ___UNSAT___, then the constraint can be removed and we can continue to the next constraint.
"""

# %%
def musx(soft_constraints, hard_constraints=[], verbose=False):
    """
        A naive pure-CP deletion-based MUS algorithm

        Will repeatedly solve the problem with one less constraint
        For normally-sized models, this will be terribly slow.

        Best is to use this only on constraints that do not support
        reification/assumption variables (e.g. some global constraints
        with expensive decompositions).
        For those constraints that do support reification, see musx_assum()
    """
    if len(soft_constraints) == 0:
        return []

    # small optimisation:
    # order so that constraints with many variables are tried first
    # this will favor MUS with few variables per constraint,
    # and will remove large constraints earlier which may speed it up
    soft_constraints = sorted(soft_constraints, key=lambda c: -len(get_variables(c)))

    # small optimisation: pre-flatten all constraints once
    # so it needs not be done over-and-over in solving
    hard = flatten_constraint(hard_constraints) # batch flatten
    soft = [flatten_constraint(c) for c in soft_constraints]

    if Model(hard+soft).solve():
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    mus_idx = [] # index into 'soft_constraints' that belong to the MUS

    # init solver with hard constraints
    m_base = Model(hard)
    for i in range(len(soft_constraints)):
        # add all other remaining (flattened) constraints
        s_without_i = CPM_ortools(m_base)
        s_without_i += soft[i+1:] 

        if s_without_i.solve():
            # with all but 'i' it is SAT, so 'i' belongs to the MUS
            if verbose:
                print("\tSAT so in MUS:", soft_constraints[i])
            mus_idx.append(i)
            m_base += [soft[i]]
        else:
            # still UNSAT, 'i' does not belong to the MUS
            if verbose:
                print("\tUNSAT so not in MUS:", soft_constraints[i])

    # return the list of original (non-flattened) constraints
    return [soft_constraints[i] for i in mus_idx]

# %%
"""
## Example: Solving with assumptions

The previous algorithm is terribly slow on larger instances, where the solver has to be re-instantiated at every iteration. Introducing assumptions ensures you can activate or turn-off some constraints. 

For instance, with assumptions b1, b2 and b3:

    b1 -> (x > 0)
    b2 -> (x < 2)
    b3 -> (x > 3)
    
If we solve with assumptions b1 and b2 set to true, that means `(x > 0)` and `(x < 2)` must be true. However the solver will set b3 to False, therefore "deactivating" the constraint. The solver will be able to find an assignment that satisfies all constraints. 
"""

# %%
x = intvar(0, 9, name="x")
b1, b2, b3 = boolvar(3, name='b')

m = CPM_ortools(Model(
        b1.implies(x > 0),
        b2.implies(x < 2),
        b3.implies(x > 3)
    ))

# Modify underlying assumptions to test turning constraints on/off
assert (m.solve(assumptions=[b1, b2]) is True)
assert (m.solve(assumptions=[b1, b2, b3]) is False)

# %%
def musx_assum(soft_constraints, hard_constraints=[], verbose=False):
    """
        An CP deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        This approach assumes that each soft_constraint supports
        reification, that is that BoolVar().implies(constraint)
        is supported by the solver or can be efficiently decomposed
        (which may not be the case for certain global constraints)
    """
    if len(soft_constraints) == 0:
        return []

    # init with hard constraints
    assum_model = Model(hard_constraints)

    # make assumption indicators, add reified constraints
    ind = BoolVar(shape=len(soft_constraints), name="ind")
    for i,bv in enumerate(ind):
        assum_model += [bv.implies(soft_constraints[i])]
    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    # make solver once, check that it is unsat and start from core
    assum_solver = CPM_ortools(assum_model)
    if assum_solver.solve(assumptions=ind):
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []
    else:
        # unsat core is an unsatisfiable subset
        mus_vars = assum_solver.get_core()
        if verbose:
            assert (not assum_solver.solve(assumptions=mus_vars)), "core is SAT!?"
        
    # now we shrink the unsatisfiable subset further
    i = 0 # we wil dynamically shrink mus_vars
    while i < len(mus_vars):
        # add all other remaining literals
        assum_lits = mus_vars[:i] + mus_vars[i+1:]

        if assum_solver.solve(assumptions=assum_lits):
            # with all but 'i' it is SAT, so 'i' belongs to the MUS
            if verbose:
                print("\tSAT so in MUS:", soft_constraints[indmap[mus_vars[i]]])
            i += 1
        else:
            # still UNSAT, 'i' does not belong to the MUS
            if verbose:
                print("\tUNSAT so not in MUS:", soft_constraints[indmap[mus_vars[i]]])
            # overwrite current 'i' and continue
            # could do get_core but then have to check that mus_vars[:i] match
            mus_vars = assum_lits


    # return the list of original (non-flattened) constraints
    return [soft_constraints[indmap[v]] for v in mus_vars]

# %%
"""
# Tutorial Example
"""

# %%
x = intvar(0, 3, shape=4, name="x")
# circular "bigger then", UNSAT
mus_cons = [
    x[0] > x[1], 
    x[1] > x[2],
    x[2] > x[0],
    
    x[3] > x[0],
    (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
]
          
print(musx_assum(mus_cons, verbose=True))

# %%
"""
# MUSX Limitations

MUSX doesn't guarantee optimality. It does not guarantee to find the smallest one (*subset minimality* of the MUS) and since it does not allow for sepcifying weigths, there is no guarantee cost-optimality.
"""

# %%
m = unsat_int_model()

## Extract unsatisfiable core
print(musx_assum(m.constraints))