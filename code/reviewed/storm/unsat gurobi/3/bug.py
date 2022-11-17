from cpmpy import *

file = "mutant_72"
solver = "gurobi"
fileM = "almostMinimized"

m = Model().from_file(fileM)
sol = m.solve(solver=solver)#, solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(fileM)
soll = m2.solve(solver="minizinc:gurobi")#, solution_limit=100)
print(soll)
print(m2.status())

print(m2)