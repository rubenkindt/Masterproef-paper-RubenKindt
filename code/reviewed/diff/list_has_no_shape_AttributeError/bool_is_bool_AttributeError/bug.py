from cpmpy import *
from simplificationWithError import simplification

file ="stable_marriage1665053345939897"
file = "almostMinimized"
solver = "ortools"

m=Model().from_file(file)
m.solve(solver="gurobi")
m.solve(solver="ortools")
