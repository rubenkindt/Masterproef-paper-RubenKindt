from mus import *
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables_model
from simplification import *

file = "bibd16650522531582546"
file = "almostMinimized"
file = "minimalInput"
solver = "gurobi"

m = Model().from_file(file)
sol = m.solve(solver=solver)


m.solve(solver="minizinc:gurobi")
print(m.status())
m.solve(solver="gurobi")
print(m.status())
m.solve(solver="minizinc:ortools")
print(m.status())
m.solve(solver="ortools")
print(m.status())