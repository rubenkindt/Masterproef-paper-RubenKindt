from mus import *
from cpmpy import *

file = "_Modif"
file = "minimal"
solver = "minizinc:ortools"

m = Model().from_file(file)
sol = m.solve(solver=solver)
print(m.status())

m2 = Model().from_file(file)
soll = m2.solve(solver="gurobi") # ortools
print(soll)
print(m2.status())

f = str(m2.constraints)
print(f)