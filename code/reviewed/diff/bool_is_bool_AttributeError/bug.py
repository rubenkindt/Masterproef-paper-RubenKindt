from cpmpy import *
from simplificationWithError import simplification

file ="stable_marriage1665053345939897"
solver = "ortools"

m=Model().from_file(file)
m=simplification(m.constraints, solver=solver)
m=Model(m)
m.to_file("minimized")
m.solve(solver="gurobi")
m.solve(solver="ortools")
