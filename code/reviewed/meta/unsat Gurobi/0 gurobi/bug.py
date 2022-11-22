from mus import *
from cpmpy import *

file = "_Modif"
file = "almostedMinimized"
solver = "gurobi"

m = Model().from_file(file) # randomize constraints and it becomes sat
# m = Model(m.constraints)
# m = mus_naive(m.constraints, solver=solver)
# m = Model(m)
# m.to_file("2")

sol = m.solve(solver=solver)
print(sol)
print(m.status())

m2 = Model().from_file(file)
soll = m2.solve(solver="ortools")#, solution_limit=100)
print(soll)
print(m2.status())

f = str(m2.constraints)
print(f)