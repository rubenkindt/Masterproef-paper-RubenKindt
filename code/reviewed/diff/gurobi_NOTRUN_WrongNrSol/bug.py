from cpmpy import *
from simplification import simplification

file = "part21665053920191321"
# file = "minimized"
solver = "gurobi"

m = Model().from_file(file)
m = simplification(m.constraints, solver=solver)
m=Model(m)
m.to_file("minimized")
sol = m.solveAll(solver="gurobi", solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(file)
sol = m2.solveAll(solver="ortools", solution_limit=100)
print(sol)
print(m2.status())
print(m2.constraints)

