from musx2 import *
from cpmpy import *


file = "mamas_age1665053190967175"
file = "almostMinimized"

m = Model().from_file(file)
# m=mus(m.constraints,solver="gurobi")
# m=mus(m)
# m.to_file("2")
sol = m.solve(solver="gurobi")
print(m)
print(m.status())
