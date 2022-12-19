from cpmpy import *

i = intvar(lb=-10, ub=10)
m = Model()
m += pow(i, 2) == 9

m.solve(solver="gurobi") # GurobiError



