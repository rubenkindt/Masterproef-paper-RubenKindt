from cpmpy import *

file = "_Modif"
solver = "ortools"

m = Model().from_file(file)
sol = m.solve(solver=solver, time_limit = 60*5)#, solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(file)
soll = m2.solve(solver="minizinc:ortools")#, solution_limit=100)
print(soll)
print(m2.status())

f = str(m2.constraints)
print(f)