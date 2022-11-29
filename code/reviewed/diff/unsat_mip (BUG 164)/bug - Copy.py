from simplification import simplification
from cpmpy import *
import random
from cpmpy.transformations.get_variables import get_variables, get_variables_model
file = "2"
# file = "minimized"

m = Model().from_file(file)
random.shuffle(m.constraints)
m = Model(m.constraints)
m.solve(solver="minizinc:mip") # same with osicbc, float, coinbc
print(m.status())

# m = simplification(m.constraints, solver="minizinc:mip")
# m = Model(m)
# m = m.to_file("2")

m = Model().from_file(file)
s = SolverLookup.get("minizinc:chuffed", m)
print("".join(map(str, s.mzn_model._code_fragments)))
# s.solve()
print(s.status())

for i in get_variables_model(m):
    print(i + ".lb="+str(i.lb) +",ub="+str(i.ub)+"name="+str(i.name))
