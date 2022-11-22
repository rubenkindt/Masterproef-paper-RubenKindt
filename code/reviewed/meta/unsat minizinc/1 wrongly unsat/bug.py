from mus import *
from cpmpy import *

file = "_Modif"
file = "minimal"
solver = "minizinc:gurobi"

m = Model().from_file(file)
sol = m.solve(solver=solver)
print(sol)
print(m.status())

m2 = Model().from_file(file)
soll = m2.solve(solver="ortools") # or gurobi
print(soll)
print(m2.status())

f = str(m2.constraints)
print(f)