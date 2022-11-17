from cpmpy import *

file = "mutant_94"
solver = "minizinc:ortools"
fileM = "almostMinimized"

m = Model().from_file(fileM)
sol = m.solve(solver=solver)#, solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(fileM)
soll = m2.solve(solver="ortools")#, solution_limit=100)
print(soll)
print(m2.status())

print(m2)