from cpmpy import *

file = "knights_path1665053474026513"

m = Model().from_file(file)
m.solve("minizinc:chuffed")
print(m.status())

m = Model().from_file(file)
m.solve("minizinc:ortools")
print(m.status())

m = Model().from_file(file)
m.solve("ortools")
print(m.status())