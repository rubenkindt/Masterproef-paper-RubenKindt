# %%
import numpy as np
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools

from IPython.display import display, HTML # display some titles

# %%
"""
# Explaining Unsatisfiability

In the previous notebookm we have seen that when errors can occur during modeling of constrained problems one can debug these models by extracting the unsatisfiable constraints. 
In that case, a minimal subset of unsatisfiable constraints (MUS, a.k.a Minimal Unsatisfiable Subset) forms an explanation of unsatisfiability [1]. 

MUS extraction algorithms such as MUSX [2] from  doesn't guarantee optimality. It does not guarantee to find the smallest one (*subset minimality* of the MUS) and since it does not allow for sepcifying weigths, there is no guarantee cost-optimality.

A ___Smallest Minimal Unsatisfiable Subset (SMUS)___ [3] forms the ___simplest/easiest___ explanation of UNSAT when all cosntraints are considered equally important, i.e. ___unweighted___ constraints.

When weights are assigned to constraint, an ___Optimal Constrained Unsatisfiable Subset (OCUS)___ [4] of the constraints will be the ___simplest/easiest___ explanation of UNSAT.

In this notebook, we consider an algorithm for computing OCUSs to illustrate the effect of assumptions, incrementality and multiple-solvers.

__OCUS__ algorithm is based on the implicit hitting set duality between Minimum Unsatisfiable Subsets (MUS) and Minimum Correction Subsets (MCS) .

We apply the MUS-extraction technique on a toy an int-variable CP model.

### ___References:___

[1] Reiter, R. (1987). A theory of diagnosis from first principles. Artificial intelligence, 32(1), 57-95.

[2] Liffiton, M. H., & Sakallah, K. A. (2008). Algorithms for computing minimal unsatisfiable subsets of constraints. Journal of Automated Reasoning, 40(1), 1-33.

[3] Ignatiev, A., et al. "Smallest MUS extraction with minimal hitting set dualization." International Conference on Principles and Practice of Constraint Programming. Springer, Cham, 2015.

[4] Gamba, E., Bogaerts, B., & Guns, T. (8 2021). Efficiently Explaining CSPs with Unsatisfiable Subset Optimization. In Z.-H. Zhou (Red), Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21 (bll 1381–1388). doi:10.24963/ijcai.2021/191.
"""

# %%
"""
## Unsatisfiable int model
"""

# %%
def unsat_int_model():
    # weights correspond to prefrences or difficutly of a constraint when
    # computing an OCUS
    # SMUS: assign unit weights 
    # weights = [1] * 10
    weights = [10, 10, 10, 1, 1, 40, 20, 20, 20, 1]

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

    return m, weights

# %%
"""
# Optimal Unsatisfiable Subsets

OCUS, or Optimal Constrained Unsatisfiable SUbsets, is a cost-optimal unsatisfiable subset that allows for a structural constraint.
The OCUS algorithm takes advantage of the implicit hitting set duality between Minimum Correction Subsets and Minimum Unsatisfiable Subsets to efficiently compute an OCUS. 

## 1. OCUS without assumptions 
"""

# %%
def OCUS(soft, soft_weights, hard=[], solver='ortools', verbose=1):
    ## Mip model
    if Model(hard+soft).solve():
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    # decision variables of hitting set problem
    hs_vars = boolvar(shape=len(soft), name="hs_vars")

    hs_model = Model(
        # Objective: min sum(w_l * x_l)
        minimize=sum(soft_weights[id] * var for id, var in enumerate(hs_vars))
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_model)

    while(True):
        # Get hitting set
        hittingset_solver.solve()
        hs_ids = np.where(hs_vars.value() == 1)[0]

        hs_soft = [soft[i] for i in hs_ids]

        # instantiate model every loop
        if not Model(hard+hs_soft).solve():
            return hs_soft

        # compute complement of model in formula F
        C = hs_vars[hs_vars.value() != 1]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

# %%
"""
## OCUS with assumptions 

The OCUS algorithm only requires little modifications in order to add assumptions.
"""

# %%
def OCUS_assum(soft, soft_weights, hard=[], solver='ortools', verbose=1):
    assert len(soft) == len(soft_weights), f"#{soft=} = #{soft_weights=} should be the same"
    # init with hard constraints
    assum_model = Model(hard)

    # make assumption indicators, add reified constraints
    ind = BoolVar(shape=len(soft), name="ind")
    for i,bv in enumerate(ind):
        assum_model += [bv.implies(soft[i])]

    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    assum_solver = SolverLookup.lookup(solver)(assum_model)

    if assum_solver.solve(assumptions=ind):
        return []

    ## 
    hs_model = Model(
        # Objective: min sum(x_l * w_l)
        minimize=sum(x_l * w_l for x_l, w_l in zip(ind, soft_weights))
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_model)

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs = ind[ind.value() == 1]

        if not assum_solver.solve(assumptions=hs):
            return soft[ind.value() == 1]

        # compute complement of model in formula F
        C = ind[ind.value() != 1]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

# %%
"""
The next cell shows that the previously found MUS was not subset-optimal and also not cost-optimal.
"""

# %%
model, weights = unsat_int_model()

# Adapt weights to see how the optimality guarantee affects
# the extraction of an Optimal Unsatisfiable Subset.
ocus = OCUS(soft=model.constraints,soft_weights=weights)
print(ocus)

# %%
"""
## MUSX
"""

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
## Comparison with deletion-based mus extraction algorithm 

Since MUSx does not guarantee to find the smallest one (*subset minimality* of the MUS) and since it does not allow for sepcifying weigths, there is no guarantee cost-optimality.
"""

# %%
model, weights = unsat_int_model()

## Extract unsatisfiable core
print(musx_assum(model.constraints))