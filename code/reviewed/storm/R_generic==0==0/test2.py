from cpmpy import *
from cpmpy.transformations.flatten_model import *


file ="minimized"

m2 = Model().from_file(file)
print(m2.constraints)
m2.solve(solver="minizinc:chuffed")
print(m2.status())

m = Model().from_file(file)
m.solve(solver="ortools")
m.status()
print(m.status())
