from simplification import simplification
from cpmpy import *

file = "exodus16650527724432807"
# file = "minimized"

m = Model().from_file(file)
m.solve(solver="minizinc:mip") # same with osicbc, float, coinbc
print(m.status())

# m = simplification(m.constraints, solver="minizinc:mip")
# m = Model(m)
# m = m.to_file("2")