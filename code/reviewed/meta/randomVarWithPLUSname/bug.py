from cpmpy import *

i = intvar(lb=3, ub=6, name="+int")
m = Model()
m += i>4
m.solve(solver="minizinc:gurobi")
