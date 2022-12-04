from cpmpy import *

X = intvar(lb=0, ub=9)
m = Model()
m += X == 3
m += ~(~(X == 3))

m.solve(solver="gurobi")
print(m.status().exitstatus.name) # UNSATISFIABLE
m.solve(solver="ortools")
print(m.status().exitstatus.name) # UNSATISFIABLE
m.solve(solver="minizinc:chuffed")
print(m.status().exitstatus.name) # FEASIBLE
