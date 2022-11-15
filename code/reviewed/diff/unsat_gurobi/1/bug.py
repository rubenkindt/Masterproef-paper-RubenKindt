from cpmpy import *


file = "bibd16650522531582546"
file = "almostMinimized"

m = Model().from_file(file)
sol = m.solve(solver="gurobi")
print(m.status())
print(m)