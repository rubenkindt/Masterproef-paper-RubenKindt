from cpmpy import *


file = "magic_square_water_retention1665052496719047"
file = "almostMinimized"

m = Model().from_file(file)
sol = m.solve(solver="gurobi")
print(m)
print(m.status())
