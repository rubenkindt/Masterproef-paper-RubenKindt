# %%
"""
# Explaining how to solve a Sudoku

This notebook covers how to generate a sequence of explanations for Constraint Satisfaction problems. Here the use case is: 

You are solving a Sudoku, but at one point, you don't know how to continue. Is there a _HINT_ button I can press to help me out and tell me which number I should write next? 

The answer is YES! 

In this notebook, we present how the reasoning behind this button works in practice. We show how to generate these ___explanations___ and how you can adapt them in order to fit your preferences.

First we load the CPMpy library:

"""

# %%
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables

import numpy as np
from time import time
from tqdm import tqdm

from IPython.display import display, HTML # display some titles

# %%
"""
## 1. Modeling the Sudoku

In practice, some puzzlers like annotate their sudoku with the numbers that cannot be selected. To keep the explanations limited,in this notebook, we model the sudoku using integer variables and consider only explanations consisting of positive assignments. 
"""

# %%
def model_sudoku(given):
    #empty cell
    e = 0
    # Dimensions of the sudoku problem
    ncol = nrow = len(given)
    n = int(ncol ** (1/2))

    # Model Variables
    cells = intvar(1,ncol,shape=given.shape,name="cells")
    
    # sudoku must start from initial clues
    facts = [cells[given != e] == given[given != e]]
    # rows constraint
    row_cons = [AllDifferent(row) for row in cells]
    # columns constraint
    col_cons = [AllDifferent(col) for col in cells.T]
    # blocks constraint
    # Constraints on blocks
    block_cons = []

    for i in range(0,nrow, n):
        for j in range(0,ncol, n):
            block_cons += [AllDifferent(cells[i:i+n, j:j+n])]

    return cells, facts, row_cons + col_cons + block_cons

# %%
"""
## Solving the Sudoku

Call the solver on the Sudoku model and extract the solution.
"""

# %%
def extract_solution(constraints, variables):
    solved = Model(constraints).solve()
    assert solved, "Model is unsatisfiable."
     
    return variables.value()

# %%
"""
Pretty printing of a sudoku grid
"""

# %%
def print_sudoku(grid):
    # Puzzel dimensions
    nrow = ncol = len(grid)
    n = int(nrow ** (1/2))

    out = ""    
    for r in range(0,nrow):
        for c in range(0,ncol):
            out += str(grid[r, c] if grid[r, c] > 0 else ' ')
            out += '  ' if grid[r, c] else '  '
            if (c+1) % n == 0 and c != nrow-1: # end of block
                out += '| '
        out += '\n'
        if (r+1) % n == 0 and r != nrow-1: # end of block
            out += ('-'*(n + 2*n))
            out += ('+' + '-'*(n + 2*n + 1)) *(n-1) + '\n'
    print(out)   

# %%
"""
## Example SUDOKU: Model and Solve

The following grid is an example of a Sudoku grid, where the given values have a value greater than 0 and the others are fixed to 0.
"""

# %%
"""
### Sample 4x4 sudoku grid
"""

# %%
e = 0
given_4x4 = np.array([
    [ e, 3, 4, e ],
    [ 4, e, e, 2 ],
    [ 1, e, e, 3 ],
    [ e, 2, 1, e ],
])

display(HTML('<h3> INPUT SUDOKU</h3>'))
print_sudoku(given_4x4)

sudoku_4x4_vars, sudoku_4x4_facts, sudoku_4x4_constraints = model_sudoku(given_4x4)

sudoku_4x4_solution = extract_solution(
    constraints=sudoku_4x4_constraints + sudoku_4x4_facts, 
    variables=sudoku_4x4_vars
)

display(HTML('<h3> SOLUTION </h3>'))
print_sudoku(sudoku_4x4_solution)

# %%
"""
### Sample 9x9 sudoku grid
"""

# %%
e = 0
given_9x9 = np.array([
    [e, e, e,  2, e, 5,  e, e, e],
    [e, 9, e,  e, e, e,  7, 3, 2],
    [e, e, 2,  e, e, 9,  e, 6, e],

    [2, e, e,  e, e, e,  4, e, 9],
    [e, e, e,  e, 7, e,  e, e, e],
    [6, e, 9,  e, e, e,  e, e, 1],

    [e, 8, e,  4, e, e,  1, e, e],
    [e, 6, 3,  e, e, e,  e, 8, e],
    [e, 2, e,  6, e, 8,  e, e, e]])

display(HTML('<h3> INPUT SUDOKU</h3>'))
print_sudoku(given_9x9)

sudoku_9x9_vars, sudoku_9x9_facts, sudoku_9x9_constraints = model_sudoku(given_9x9)

sudoku_9x9_solution = extract_solution(
    constraints=sudoku_9x9_constraints + sudoku_9x9_facts, 
    variables=sudoku_9x9_vars
)

display(HTML('<h3> SOLUTION </h3>'))
print_sudoku(sudoku_9x9_solution)

# %%
"""
## Explanations for SUDOKU: 

To explain a full sudoku from the givens to the solution, we generate a sequence of intermediate expalnation steps.
Every explanation step is characterized by an *interpretation*, which corresponds to current status of the grid. 
The initial state of the gris is called the ***initial interpretation***, and the solution is also known as the ***end interpretation***.

Every explanation step uses subset of the problem constraints and part of the interpretation in order to derive one or multiple new numbers. 

1. __C__' ⊂ __C__ A subset of the problem constraints (alldifferents on columns, rows and blocks).

2. __I__' ⊂ __I__ A partial interpretation

    - In the Sudoku case this corresponds to the numbers filled in the grid at the current explanation step (givens, and newly derived numbers).

3. __N__ A newly found number to fill in the grid.

Therefore at every step, the number 

To compute such explanations, we take advantage of the link between non-redundant explanations and Minimal Unsatisfiable Subsets introduced in [1]. 


[1] Bogaerts, B., Gamba, E., & Guns, T. (2021). A framework for step-wise explaining how to solve constraint satisfaction problems. Artificial Intelligence, 300, 103550.

"""

# %%
class SMUSAlgo:

    def __init__(self, soft, hard=[], solvername="ortools", hs_solvername="gurobi", disable_corr_enum=True):
        """
            Smallest Minimal Unsatisfiable Subsets (SMUS) for CSP explanations [1,2]

            SMUS relies on the implicit hitting set duality between
            MUSes and MCSes for a given formula F [2, 3]:

                - A set S \subseteq of F is an MCS of F iff it is a minimal hitting set
                 of MUSs(F).
                - A set S \subseteq F is a MUS of F iff it is a minimal hitting set
                 of MCSs(F).

            Builds a MIP model for computing minimal (optimal) hitting sets. Repeatedly
            checks for satisfiability until UNSAT, which means an SMUS is found.

            If SAT, and sat solver computes a model. The complement w.r.t F is
            computed and added as a new set-to-hit.

            Args
            ----

                hard : Hard constraints must be included in the MUS

                soft : Soft Constraints can be selected by the MUS algorithm

            [1] Ignatiev, A., Ignatiev, A., Previti, A., Liffiton, M. and Marques-Silva, J (2015).  
            Smallest MUS extraction with minimal hitting set dualization. 
            International Conference on Principles and Practice of Constraint Programming. 
            Springer, Cham, 2015.

            [2] Gamba, E., Bogaerts, B., & Guns, T. (8 2021). Efficiently Explaining CSPs
            with Unsatisfiable Subset Optimization. In Z.-H. Zhou (Red), Proceedings of the
            Thirtieth International Joint Conference on Artificial Intelligence,
            IJCAI-21 (bll 1381–1388). doi:10.24963/ijcai.2021/191.

            [3] Liffiton, M. H., & Sakallah, K. A. (2008). Algorithms for computing minimal
            unsatisfiable subsets of constraints. Journal of Automated Reasoning, 40(1), 1-33.

            [4] Reiter, R. (1987). A theory of diagnosis from first principles.
            Artificial intelligence, 32(1), 57-95.
        """
        self.soft = cpm_array(soft)
        self.hard = hard
        
        self.assum_vars = boolvar(len(soft))

        self.solver = SolverLookup.get(solvername)
        self.solver += self.hard
        self.solver += self.assum_vars.implies(self.soft)
        
        self.maxsat_solver = SolverLookup.get(solvername)
        self.maxsat_solver += self.hard
        self.maxsat_solver += self.assum_vars.implies(self.soft)
        self.maxsat_solver.maximize(sum(self.assum_vars))

        # Hitting Set MODEL is described by:
        #     - x_l={0,1} if assumption variable is inside the hitting set (1) or not (0).
        #     - c_lj={0,1} is 1 (0) if the literal l is (not) present in the set-to-hit j.
        # Subject to:
        #     (1) sum(x_l * c_lj) >= 1 for all hitting sets j.
        #         = The hitting set must hit all sets-to-hit.
        self.hs_solver= SolverLookup.get(hs_solvername)
        self.hs_solver += sum(self.assum_vars) >= 1
        # Objective:
        #         min sum(x_l) 
        self.hs_solver.minimize(sum(self.assum_vars))

        self.mus = None
        self.disable_corr_enum = disable_corr_enum
        
    def iterate(self, n=1):
        '''
            SMUS-Core computes n iterations of the algorithm.
        '''
        for _ in range(n):
            assert self.hs_solver.solve()
            h = self.assum_vars.value() == 1
            
            if not self.solver.solve(assumptions=self.assum_vars[h]):
                # UNSAT subset, return
                self.mus = set(self.soft[self.assum_vars.value()])
                return
                
            # Find disjunctive sets
            mcses = self.corr_enum(h)

            for mcs in mcses:
                self.hs_solver += sum(self.assum_vars[mcs]) >= 1

    def corr_enum(self, h):
           
        mcses = []
        hp = np.array(h)

        sat, ss = self.grow(hp)
        
        if self.disable_corr_enum:
            return [~ss]

        while(sat):
            mcs = ~ss
            mcses.append(mcs)
            hp = hp | mcs
            sat, ss = self.grow(hp)

        return mcses            
    
    def grow(self, h):
#         from time import time
        start = time()
        sat = self.maxsat_solver.solve(assumptions=self.assum_vars[h])

        return sat, self.assum_vars.value() == 1

# %%
"""
## Explanation
This part of the notebook finds the smallest step to take in solving the sudoku. <br>
"""

# %%
def split_ocus(given, vars_to_expl, constraints, verbose=False):
    """
        Split ocus first propagates the given variable-value assignments and 
        constraints to find a satisfying solution. 
        split_ocus keep track of a priority queue (PQ) of SMUS extractors 
        for every value assignment that needs to be explained. 
        The PQ is sorted using the size of the current hitting set of every 
        variable. Therefore, the element at the top of the PQ is the most likely
        to lead to an small explanation. 
        At every iteration of the split_ocus algorithm:
        
            1. The element at the top of the PQ is popped.
            2. One iteration in the SMUS algorithm is done.
                - Its hitting set cost is updated 
            3. The element is pushed by into the list 
        
        Args:
        - given: dict variable-value
        - vars_to_expl: collection of variables for which you want a potential
        explanation.
        - constraints: Reified-problem constraints
        

    """
    facts = [var == val for var, val in given.items()]
    assert Model(facts + constraints).solve(), "Model should be SAT!"
    sol = {var : var.value() for var in vars_to_expl}
    
    ## priority queue (PQ) of SMUS extractors for every value assignment
    ## that needs to be explained
    pq = [(var,0,SMUSAlgo(soft=facts + constraints, hard=[var != sol[var]])) 
          for var in vars_to_expl]
    
    i = 0
    while 1:
        var, obj_val, smus_algo = pq.pop(0)
        if verbose:
            print(f"\rContinuing computation on SMUS with obj {obj_val}", end="\t"*5)
        # pbar.set_description(f"Best objective_score: {obj_val}")
        if smus_algo.mus is not None:
            E = smus_algo.mus & set(facts)
            S = smus_algo.mus & set(constraints)
            N = [var == sol[var]]
            return (E,S,N)
        # Algo has not found a solution yet, continue
        smus_algo.iterate()
        pq.append((var,smus_algo.hs_solver.objective_value(),smus_algo))
        pq.sort(key=lambda x : x[1])

# %%
"""
## Stepwise explanations

For this, we build upon out work on stepwise explanations of SAT problems [2].
"""

# %%
def explain_full_sudoku(given, vars_to_expl, constraints, verbose=False):
    facts = [var == val for var, val in given.items()]
    assert Model(facts + constraints).solve(), "Model should be SAT!"
    sol = {var : var.value() for var in vars_to_expl}

    remaining_vars_to_expl = set(vars_to_expl)
    
    explanation_sequence = []
    
    while(len(remaining_vars_to_expl) > 0):
        E, S, N = split_ocus(given, remaining_vars_to_expl, constraints, verbose=verbose)
        facts += N
        var_N = get_variables(N)[0]
        given[var_N] = sol[var_N]

        remaining_vars_to_expl -= set(get_variables(N))
        print(f"\n\nFacts:\n\t{E=}\nConstraints:\n\t{S=},\n\t => {N=}\n")
        
        explanation_sequence.append((E,S,N))
    
    return explanation_sequence

# %%
"""
## Generate a sample explanation for given Sudoku
"""

# %%
given = given_4x4

cells, facts, constraints = model_sudoku(given)

clues = {var : val for var, val in zip(cells.flatten(), given.flatten()) if val != e}
vars_to_expl = set(get_variables(constraints)) - set(clues.keys())

start = time()
E,S,N = split_ocus(clues, vars_to_expl, constraints, verbose=True)

print(f"\n\nFacts:\n\t{E=}\nConstraints:\n\t{S=},\n\t => {N=}\n")

# %%
"""
## Generate the full explanation sequence for given Sudoku
"""

# %%
given = given_4x4

cells, facts, constraints = model_sudoku(given)

clues = {var : val for var, val in zip(cells.flatten(), given.flatten()) if val != e}
vars_to_expl = set(get_variables(constraints)) - set(clues.keys())

start = time()
explain_full_sudoku(clues, vars_to_expl, constraints, verbose=True)

# %%
