from musx2 import *
from cpmpy import *


file = "magic_square_water_retention16650524967205071"
file = "almostMinimized"

m = Model().from_file(file)
# m=mus(m.constraints,solver="gurobi")
# m=mus(m)
# m.to_file("2")
sol = m.solve(solver="gurobi")
print(m)
print(m.status())
