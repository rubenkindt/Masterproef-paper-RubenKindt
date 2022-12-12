from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
from cpmpy.transformations.get_variables import print_variables

X = intvar(lb=0, ub=9, name='X')
m = Model()
m += X == 3
m += ~(~(X == 3))

m.solve(solver="gurobi")
print(m.status().exitstatus.name) # UNSATISFIABLE

# mf = flatten_model(m)
print_variables(m)
print(m)

m.solve(solver="ortools")
print(m.status().exitstatus.name) # UNSATISFIABLE
m.solve(solver="minizinc:chuffed")
print(m.status().exitstatus.name) # FEASIBLE
