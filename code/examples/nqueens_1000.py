# %%
"""
## Solving n-queens with a modern C(S)P solver

This notebook was created for the CSP lecture of the 'Intro to AI' lecture by Tias Guns, based on the famous Russell and Norvig book.

Somebody had written on a slide that CSP solving with backtracking search, arc consistency and a good variable ordering makes solving n-queens for N=1000 'feasible'.

So, this notebook checks how 'feasible' n-queens for different N is for a modern CP solver like or-tools CP-SAT.
(technical note: we first tuned the hyper parameters on N=100 so that the solver is performing at its best)
"""

# %%
"""
N-queens problem in CPMpy

CSPlib prob054

Problem description from the numberjack example:
The N-Queens problem is the problem of placing N queens on an N x N chess
board such that no two queens are attacking each other. A queen is attacking
another if it they are on the same row, same column, or same diagonal.
"""

# load the libraries
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools

def nqueens(N):
    # Variables (one per row)
    queens = intvar(1,N, shape=N, name="queens")

    # Constraints on columns and left/right diagonal
    m = Model([
        AllDifferent(queens),
        AllDifferent([queens[i] + i for i in range(N)]),
        AllDifferent([queens[i] - i for i in range(N)]),
    ])
    
    return (m, queens)

def nqueens_solve(N, prettyprint=True):
    (m, queens) = nqueens(N)
    
    # tuned params: {'cp_model_probing_level': 0, 'linearization_level': 0, 'symmetry_level': 0}
    s = CPM_ortools(m)
    if s.solve(cp_model_probing_level=0, linearization_level=0, symmetry_level=0):
        print(s.status())
        
        if prettyprint:
            # pretty print
            line = '+---'*N+'+\n'
            out = line
            for queen in queens.value():
                out += '|   '*(queen-1)+'| Q '+'|   '*(N-queen)+'|\n'
                out += line
            print(out)
    else:
        print("No solution found")
    
N = 4
nqueens_solve(N)

# %%
for N in [8,25,100,200,500,1000]:
    print("Solving",N,"queens...")
    prettyprint = (N < 100)
    nqueens_solve(N, prettyprint)
    print() # empty line

# %%


# %%
# param tune to find the fastest solver parameters
(m, x) = nqueens(100)

m.solve()
base_runtime = m.status().runtime
print("Runtime with default params", base_runtime)

from cpmpy.solvers import CPM_ortools, param_combinations

all_params = {'cp_model_probing_level': [0,1,2,3],
              'linearization_level': [0,1,2],
              'symmetry_level': [0,1,2]}

configs = [] # (runtime, param)
for params in param_combinations(all_params):
    s = CPM_ortools(m)
    print("Running", params, end='\r')
    s.solve(time_limit=base_runtime*1.05, **params)
    configs.append( (s.status().runtime, params) )
    base_runtime = min([base_runtime, s.status().runtime])

best = sorted(configs)[0]
print("\nFastest in", round(best[0],4), "seconds, config:", best[1])

# %%
