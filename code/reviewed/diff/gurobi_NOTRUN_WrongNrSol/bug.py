from cpmpy import *

file = "part21665053920191321"

m = Model().from_file(file)
sol = m.solveAll(solver="gurobi", solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(file)
sol = m2.solveAll(solver="ortools", solution_limit=100)
print(sol)
print(m2.status())
