# from musx2 import *
from cpmpy import *
# from simplification import *

file = "17b16650534325711741"

m = Model().from_file(file)
m.solve(solver="gurobi")
print(m.status())

# m = simplification(m.constraints, solver="minizinc:mip")
# m = Model(m)
# m = m.to_file("2")