# %%
from cpmpy import *

# %%
from numpy.random import randint
import numpy as np
from numpy.linalg import norm

INFTY = 10000

# %%
"""
## Knapsack problem
"""

# %%
#knapsack problem
def get_knapsack_problem(N=8, capacity=35):
    np.random.seed(0)
    
    items = boolvar(shape=N, name="items")

    values = randint(0,10,size=N)
    weights = randint(0,10, size=N)

    model = Model(maximize=sum(items * values))
    model += sum(items * weights) <= capacity
    
    return model, (items, values, weights, capacity)

# %%
model, (items, values, weights, capacity) = get_knapsack_problem()
assert model.solve()
print("Objective value:",model.objective_value())
print("Used capacity:", sum(items.value() * weights))

print(f"{values = }")
print(f"{weights = }")
print(f"{capacity = }")

items.value()

# %%
# User query
# "I want my solution to really contain item 1 and 2"
model += all(items[[1,2]])
assert model.solve()

x_d = items.value()
print("Objective value:",model.objective_value())
print("Used capacity:", sum(x_d * weights))

x_d

# %%
"""
## Inverse optimize
"""

# %%
def inverse_optimize(SP, c, x, x_d, keep_static=None):
    
    # Decision variable for new parameter vector
    d = intvar(0,INFTY, shape=len(x_d), name="d")

    # create the master problem
    MP = SolverLookup.get("gurobi")
    MP.minimize(norm(c-d,1))
    MP += SP.constraints
    
    if keep_static is not None:
        MP += d[keep_static] == c[keep_static]

    while MP.solve():
        # find new d
        new_d = d.value()
        print(f"{new_d = }")
        
        SP.maximize(sum(new_d * x))
        SP.solve()

        if sum(new_d * x_d) >= sum(new_d * x.value()):
            # solution is optimal
            break

        MP += sum(d * x_d) >= sum(d * x.value())
    return new_d

SP, (x, values, weights,_) = get_knapsack_problem()

print("Original values:", values)
keep_static = [0,3,4,5,6,7]
inverse_optimize(SP, values, x, x_d, keep_static)

# %%


# %%
