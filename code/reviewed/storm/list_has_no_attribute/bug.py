from cpmpy import *

file = "knights_path1665053474026513"

m = Model().from_file(file)
m.solve("gurobi")
print(m.status())
