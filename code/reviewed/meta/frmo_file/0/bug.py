from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

solver = "ortools"

file = "_Modif"
m = Model().from_file(file)
m.solve(solver=solver)
print(m.status())


m2 = Model(m.constraints)
if solver == ("minizinc:ortools"):
    m2 = flatten_model(m2)
m2.solve(solver=solver)
print(m2.status())