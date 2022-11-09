from cpmpy import *

file ="stable_marriage1665053345939897"
m=Model().from_file(file)
m.solve(solver="ortools")
m.solve(solver="gurobi")