# %%
from cpmpy import *
import numpy as np

# %%


# %%
x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

while m.solve():
    print(x.value())
    m += ~all(x == x.value()) # block solution

# %%


# %%
# a diversity measure, hamming distance
def hamm(x, y):
    return sum(x != y)

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

store = []
while m.solve():
    print(len(store), ":", x.value())
    m += ~all(x == x.value()) # block solution
    store.append(x.value())
    # maximize number of elements that are different
    m.maximize(sum([hamm(x, sol) for sol in store]))


# %%


# %%
import time
from cpmpy.solvers import CPM_ortools

# %%
x = intvar(0,30, shape=30)
m = Model([x[i-1] < x[i] for i in range(1, len(x))])

t0 = time.time()

while m.solve():
    print(".",end="")
    m += ~all(x == x.value()) # block solution
print("time:", time.time()-t0)

# %%
x = intvar(0,30, shape=30)
m = Model([x[i-1] < x[i] for i in range(1, len(x))])

t0 = time.time()
m = CPM_ortools(m)
while m.solve():
    print(".",end="")
    m += ~all(x == x.value()) # block solution
print("time:", time.time()-t0)

# %%


# %%
x = intvar(0,3, shape=4, name="x")
# circular 'bigger then', UNSAT
mus_cons = [
    x[0] > x[1],
    x[1] > x[2],
    x[2] > x[0],
    
    x[3] > x[0],
    (x[3] > x[1]).implies(x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2]))
]


i = 0 # we wil dynamically shrink mus_vars
while i < len(mus_cons):
    # add all other remaining constraints
    assum_cons = mus_cons[:i] + mus_cons[i+1:]

    if Model(assum_cons).solve():
        # with all but 'i' it is SAT, so 'i' belongs to the MUS
        print("\tSAT so in MUS:", mus_cons[i])
        i += 1
    else:
        # still UNSAT, 'i' does not belong to the MUS
        print("\tUNSAT so not in MUS:", mus_cons[i])
        # overwrite current 'i' and continue
        mus_cons = assum_cons

# %%


# %%
x = intvar(0,3, shape=4, name="x")
# circular 'bigger then', UNSAT
mus_cons = [
    x[0] > x[1],
    x[1] > x[2],
    x[2] > x[0],
    
    x[3] > x[0],
    (x[3] > x[1]).implies(x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2]))
]


assum_model = Model()
# make assumption indicators, add reified constraints
ind = BoolVar(shape=len(mus_cons), name="ind")
for i,bv in enumerate(ind):
    assum_model += [bv.implies(mus_cons[i])]
# to map indicator variable back to soft_constraints
indmap = dict((v,i) for (i,v) in enumerate(ind))

assum_solver = CPM_ortools(assum_model)
assert (not assum_solver.solve(assumptions=ind)), "Model must be UNSAT"

# unsat core is an unsatisfiable subset
mus_vars = assum_solver.get_core()
print("UNSAT core of size", len(mus_vars))

# now we shrink the unsatisfiable subset further
i = 0 # we wil dynamically shrink mus_vars
while i < len(mus_vars):
    # add all other remaining constraints
    assum_vars = mus_vars[:i] + mus_vars[i+1:]

    if assum_solver.solve(assumptions=assum_vars):
        # with all but 'i' it is SAT, so 'i' belongs to the MUS
        print("\tSAT so in MUS:", mus_cons[i])
        i += 1
    else:
        # still UNSAT, 'i' does not belong to the MUS
        print("\tUNSAT so not in MUS:", mus_cons[i])
        # overwrite current 'i' and continue
        mus_vars = assum_vars

# %%


# %%
from marco_musmss_enumeration import SubsetSolver, MapSolver

def do_marco(model):
    sub_solver = SubsetSolver(model.constraints)
    map_solver = MapSolver(len(model.constraints))

    while True:
        seed = map_solver.next_seed()
        if seed is None:
            # all MUS/MSS enumerated
            return

        if sub_solver.check_subset(seed):
            MSS = sub_solver.grow(seed)
            yield ("MSS", [model.constraints[i] for i in MSS])
            map_solver.block_down(MSS)
        else:
            seed = sub_solver.seed_from_core()
            MUS = sub_solver.shrink(seed)
            yield ("MUS", [model.constraints[i] for i in MUS])
            map_solver.block_up(MUS)

# %%
x = intvar(0,3, shape=4, name="x")
# circular 'bigger then', UNSAT
m = Model(
    x[0] > x[1],
    x[1] > x[2],
    x[2] > x[0],
    
    x[3] > x[0],
    (x[3] > x[1]).implies(x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2]))
)


for kind, exprs in do_marco(m):
    print(kind,":")
    for e in sorted(exprs):
        print("\t", e)

# %%


# %%
x = intvar(0,3, shape=4, name="x")
# circular 'bigger then', UNSAT
mus_cons = [
    x[0] > x[1],
    x[1] > x[2],
    x[2] > x[0],
    
    x[3] > x[0],
    (x[3] > x[1]).implies(x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2]))
]
weights = np.array([2,2,2, 5,10])


assum_model = Model()
# make assumption indicators, add reified constraints
ind = BoolVar(shape=len(mus_cons), name="ind")
for i,bv in enumerate(ind):
    assum_model += [bv.implies(mus_cons[i])]
# to map indicator variable back to soft_constraints
indmap = dict((v,i) for (i,v) in enumerate(ind))

assum_solver = CPM_ortools(assum_model)
assert (not assum_solver.solve(assumptions=ind)), "Model must be UNSAT"


hitset_solver = CPM_ortools(Model(
                    minimize=sum(weights*ind)))

while(True):
    hitset_solver.solve()

    # Get hitting set
    hs = ind[ind.value() == 1]

    if not assum_solver.solve(assumptions=hs):
        print("Found Optimal US, total weight:", sum(weights[ind.value() == 1]))
        for i in (ind.value() == 1).nonzero()[0]:
            print("\t", mus_cons[i], "w=",weights[i])
        break

    # hs is satisfiable subset, hit one from complement
    C = ind[ind.value() == 0]
    hitset_solver += (sum(C) >= 1)


# %%


# %%
b = boolvar(3, name="b")
m = Model(
    b[1].implies(b[0] | b[2]),
    b[0] | b[1],
    ~b[0],
)
m.solve()

from ocus_explanations import explain_ocus
r = explain_ocus(m.constraints, verbose=True)

# %%


# %%
# FROM examples/advanced/counterfactual_explain.py
# cutting plane algorithm

def inverse_optimize(d_orig, weights, capacity, x_d, foil_idx):
    """
    Master problem: iteratively find better values for the 'd_orig' vector
    (Korikov, A., & Beck, J. C., Counterfactual Explanations via Inverse Constraint Programming (CP2021))
    """
    master_model, d, x = make_master_problem(d_orig, weights, capacity, x_d, foil_idx)
    sub_model, x_0 = make_sub_problem(d_orig, weights, capacity)

    i = 1
    while master_model.solve() is not False:
        d_star = d.value() # master solution
        if verbose:
            print(f"Iteration {i}, candidate costs: {d_star}")

        # solve subproblem
        sub_model.maximize(sum(x_0 * d_star))
        sub_model.solve()
        if verbose:
            print(f"  Is foil-based solution now optimal? {sum(d_star * x_d)} >=? {sum(d_star * x_0.value())}")
        if sum(d_star * x_d) >= sum(d_star * x_0.value()):
            return d_star # is optimal
        else:
            # add cutting plane to master
            master_model += [sum(d * x) >= sum(d * x_0.value())]
        i += 1

    raise ValueError("Master model is UNSAT!")

# %%
