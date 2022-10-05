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
model_knapsack, (items, values, weights, capacity) = get_knapsack_problem()
assert model_knapsack.solve()
print("Objective value:",model_knapsack.objective_value())
print("Used capacity:", sum(items.value() * weights))

print(f"{values = }")
print(f"{weights = }")
print(f"{capacity = }")

items.value()

# %%
"""
# Demonstrate multi-solver, with same syntax and variable sharing
"""

# %%
m_ort = SolverLookup.get("ortools", model_knapsack)
m_ort.solve()
print("\nOrtools:", m_ort.status(), ":", m_ort.objective_value(), items.value())

m_grb = SolverLookup.get("gurobi", model_knapsack)
m_grb.solve()
print("\nGurobi:", m_grb.status(), ":", m_grb.objective_value(), items.value())

# use ortools to verify the gurobi solution
m_ort += (items == items.value())
print("\tGurobi's is a valid solution according to ortools:", m_ort.solve())
