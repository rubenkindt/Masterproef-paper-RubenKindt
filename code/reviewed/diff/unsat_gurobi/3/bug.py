from cpmpy import *


file = "mamas_age1665053190967175"
file = "almostMinimized"

m = Model().from_file(file)
sol = m.solve(solver="gurobi")
print(m)
print(m.status())
