from cpmpy import *


file = "football1665051982904926"
file = "almostMinimized"

m = Model().from_file(file)
sol = m.solve(solver="gurobi")
print(m)
print(m.status())
