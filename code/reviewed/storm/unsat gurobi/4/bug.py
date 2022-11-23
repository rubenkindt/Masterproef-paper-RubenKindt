from cpmpy import *

file = "mutant_72"
solver = "gurobi"
fileM = "almostMinimized"

m = Model().from_file(fileM)

m.solve(solver="minizinc:gurobi")
print(m.status())
m.solve(solver="gurobi")
print(m.status())
m.solve(solver="minizinc:ortools")
print(m.status())
m.solve(solver="ortools")
print(m.status())