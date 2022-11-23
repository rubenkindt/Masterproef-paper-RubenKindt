from mus import *
from cpmpy import *

file = "_Modif"
file = "almostedMinimized"
solver = "gurobi"

m = Model().from_file(file)
# m = Model(m.constraints)
# m = mus_naive(m.constraints, solver=solver)
# m = Model(m)
# m.to_file("2")
m.solve(solver=solver)
print(m.status())

m2 = Model().from_file(file)
m2.solve(solver="ortools")
print(m2.status())
# print(m2)

m.solve(solver="gurobi")
print(m.status())
m.solve(solver="ortools")
print(m.status())