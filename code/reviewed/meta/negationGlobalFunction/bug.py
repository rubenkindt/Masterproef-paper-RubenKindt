from cpmpy import *

pos = intvar(lb=2, ub=5,shape=3, name="position")
m = Model()
m += AllDifferent(pos) != 0
m.solve("minizinc:chuffed")
print(m.status())

pos = intvar(lb=2, ub=5,shape=3, name="position")
m = Model()
m += AllDifferent(pos) != 0
m.solve("gurobi")
m.solve("ortools")
print(m.status())