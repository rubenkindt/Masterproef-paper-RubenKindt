from simplification import simplification
from cpmpy import *
import random

file = "2"
# file = "minimized"

m = Model().from_file(file)
random.shuffle(m.constraints)
m = Model(m.constraints)
m.solve(solver="minizinc:mip") # same with osicbc, float, coinbc
print(m.status())

m = simplification(m.constraints, solver="minizinc:mip")
m = Model(m)
m = m.to_file("2")