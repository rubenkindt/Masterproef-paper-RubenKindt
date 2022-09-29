# %%
"""
# Enumeration of Minimal Unsatisfiable Cores and Maximal Satisfying Subsets

This tutorial illustrates how to use (CPMpy) for extracting all minimal unsatisfiable
cores together with all maximal satisfying subsets. 

## Origin

The algorithm that we describe next represents the essence of the core extraction
procedure by Liffiton and Malik and independently by Previti and Marques-Silva: 

    Enumerating Infeasibility: Finding Multiple MUSes Quickly
    Mark H. Liffiton and Ammar Malik
    in Proc. 10th International Conference on Integration of Artificial Intelligence (AI)
    and Operations Research (OR) techniques in Constraint Programming (CPAIOR-2013), 160-175, May 2013. 

    Partial MUS Enumeration
    Alessandro Previti, Joao Marques-Silva in Proc. AAAI-2013 July 2013 

It illustrates the following features of CPMpy's Python-based direct access to or-tools:

1. Using assumptions to track unsatisfiable cores. 
2. Using multiple models/solvers and passing constraints between them. 


"""

# %%
import sys

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from cpmpy.solvers.ortools import CPM_ortools

# %%
"""
## 1. MAP Solver

The MapSolver is used to enumerate sets of clauses that are not already supersets of an existing unsatisfiable core and not already a subset of a maximal satisfying assignment. 

The MapSolver uses one unique atomic predicate per soft clause, so it enumerates sets of atomic predicates. 

- For each minimal unsatisfiable core, say, represented by predicates p1, p2, p5, the MapSolver contains the clause  !p1 | !p2 | !p5. 

- For each maximal satisfiable subset, say, represented by predicats p2, p3, p5, the MapSolver contains a clause corresponding to the disjunction of all literals not in the maximal satisfiable subset, p1 | p4 | p6. 

"""

# %%
def next_seed(map_solver, ind_vars, verbose=False):
    
    if map_solver.solve() is False:
            return None
    if verbose:
        print("\nSeed:", ind_vars[ind_vars.value()==1])
    return ind_vars[ind_vars.value()==1]

def block_down(map_solver, ind_vars, MSS, verbose=False):
    """Block down from a given set."""
    if verbose:
        print("\tblocking down:", any(v for v in set(ind_vars)-set(MSS)))
    map_solver += any(v for v in set(ind_vars)-set(MSS))

def block_up(map_solver, MUS, verbose=False):
    if verbose:
        print("\tblocking up:", any(~v for v in MUS))
    map_solver += any(~v for v in MUS)

# %%
"""
## 2. Subset Solver

The SubsetSolver contains a set of soft clauses (clauses with the unique indicator atom occurring negated). 

The MapSolver feeds it a set of clauses (the indicator atoms). Recall that these are not already a superset of an existing minimal unsatisfiable core, or a subset of a maximal satisfying assignment: 

- If asserting these atoms makes the SubsetSolver context ***infeasible***, then it finds a minimal unsatisfiable subset corresponding to these atoms. 
- If asserting the atoms is ***consistent*** with the SubsetSolver, then it extends this set of atoms maximally to a satisfying set. 
"""

# %%
def grow(subset_solver, ind_vars, seed, verbose=False):
    """
        Grow the satisfiable subset.
    """
    current = list(seed)
    if verbose:
        print(f"Growing ({current})")
    # Try to add the literals of seed's complement
    for i in set(ind_vars) - set(seed):
        if subset_solver.solve(assumptions=current + [i]):
            if verbose:
                print("\t\tSAT so", i, "in MSS:\n\t\t", current, "->", current + [i])
            current.append(i)
    return current

def shrink(subset_solver, ind_vars, seed, verbose=False):
    current = list(seed)
    
    # Try to add the literals of seed's complement
    i = 0 # we wil dynamically shrink mus_vars
    if verbose:
        print("\t -> Shrinking:", current)
    
    while i < len(current):
        # add all other remaining literals
        assum_lits = current[:i] + current[i+1:]

        if subset_solver.solve(assumptions=assum_lits):
            # with all but 'i' it is SAT, so 'i' belongs to the MUS
            if verbose:
                print("\t\tSAT so", current[i] ,"in MUS, keeping", )
            i += 1
        else:
            # still UNSAT, 'i' does not belong to the MUS
            if verbose:
                print("\t\tUNSAT so",current[i] , "not in MUS:", assum_lits)
            # overwrite current 'i' and continue
            # could do get_core but then have to check that mus_vars[:i] match
            current = assum_lits

    return current

# %%
"""
## Idea of the Marco MUS/MCS Algorithm

The main idea of the algorithm is to maintain two logical contexts and exchange information between the ***MapSolver*** and ***Subset Solver***.
"""

# %%
def do_marco(constraints, solvername="ortools", verbose=False): #csolver, map):
    """
        Basic MUS/MCS enumeration, as a simple example.
        
        Warning: all constraints in 'mdl' must support reification!
        Otherwise, you will get an "Or-tools says: invalid" error.
    """
    # SUBSET solver
    ## Adding indicator variables
    ind_vars = BoolVar(shape=len(constraints))
    idcache = dict((v,i) for (i,v) in enumerate(ind_vars))

    ## Reifying constraints with indicator variables
    mdl_reif = Model([ind_vars[i].implies(con) for i,con in enumerate(constraints)])
    subset_solver = SolverLookup.lookup(solvername)(mdl_reif)
    if verbose:
        print("Reifying model")
        print(mdl_reif, "\n")

    # MAP solver
    map_solver = SolverLookup.lookup(solvername)(Model([]))

    while True:
        seed = next_seed(map_solver, ind_vars, verbose=verbose)
        if seed is None:
            return

        if subset_solver.solve(assumptions=seed):
            MSS = grow(subset_solver, ind_vars, seed, verbose=verbose)
            yield ("MSS", [constraints[idcache[i]] for i in MSS])
            block_down(map_solver, ind_vars, MSS, verbose=verbose)
        else:
            MUS = shrink(subset_solver, ind_vars, seed, verbose=verbose)
            yield ("MUS", [constraints[idcache[i]] for i in MUS])
            block_up(map_solver, MUS, verbose=verbose)

# %%
"""
## Small Unsatisfiable Model
"""

# %%

x = intvar(-9, 9, name="x")
y = intvar(-9, 9, name="y")

unsat_model = Model(
    x < 0, 
    x < 1,
    x > 2,
    (x + y > 0) | (y < 0),
    (y >= 0) | (x >= 0),
    (y < 0) | (x < 0),
    (y > 0) | (x < 0),
)
assert (unsat_model.solve() is False)

print("\nStart MUS/MSS enumeration:")

# Warning, all constraints must support reification...
# SET verbose to True for more details
for kind, exprs in do_marco(unsat_model.constraints, verbose=False):
    print(f"{kind} {exprs}")