from cpmpy import *

file ="3_jugs_regular1665052420247257"
file = "minimized"
m=Model().from_file(file)
m.solve(solver="gurobi")
print(m.status())