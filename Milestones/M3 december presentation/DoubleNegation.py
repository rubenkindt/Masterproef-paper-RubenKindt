from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

X = intvar(lb=0, ub=9, name='X')
m = Model()
m += X == 3
m += ~(~(X == 3))

m.solve(solver="gurobi")
print(m.status().exitstatus.name) # UNSATISFIABLE
m.solve(solver="ortools")
print(m.status().exitstatus.name) # UNSATISFIABLE

m.solve(solver="minizinc:chuffed")
print(m.status().exitstatus.name) # FEASIBLE

print(m)
mf = flatten_model(m)
print(mf)



