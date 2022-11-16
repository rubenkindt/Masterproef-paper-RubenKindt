from cpmpy import *

file = "_Modif"
solver = "minizinc:ortools"
fileM = "minimized"

m = Model().from_file(fileM)
sol = m.solve(solver=solver, time_limit = 60*5)#, solution_limit=100)
print(sol)
print(m.status())

m2 = Model().from_file(fileM)
soll = m2.solve(solver="ortools")#, solution_limit=100)
print(soll)
print(m2.status())

f = str(m2.constraints)
print(f)