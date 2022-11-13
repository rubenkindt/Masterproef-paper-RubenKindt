from cpmpy import *

file ="minimized"

m2 = Model().from_file(file)
print(m2.constraints)
m2.solve(solver="minizinc:chuffed")
print(m2.status())

m = Model().from_file(file)
m.solve(solver="gurobi")
m.solve(solver="ortools")
m.status()
print(m.status())
