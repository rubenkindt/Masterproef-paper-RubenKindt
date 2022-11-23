from cpmpy import *

file = "mutant_94"
solver = "minizinc:ortools"
fileM = "almostMinimized"

m = Model().from_file(fileM)
sol = m.solve(solver=solver)#, solution_limit=100)
# print(sol)
# print(m.status())

m.solve(solver="minizinc:gurobi")
print(m.status())
m.solve(solver="gurobi")
print(m.status())
m.solve(solver="minizinc:ortools")
print(m.status())
m.solve(solver="ortools")
print(m.status())