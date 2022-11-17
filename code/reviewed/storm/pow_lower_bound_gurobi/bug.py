from cpmpy import *

file = "prob005_auto_correlation16650537061803274"
file = "almostMinimized"

m = Model()
i = intvar(lb=-10, ub=10)
m += pow(i,2)==9
m.solve(solver="gurobi")
print(str(m.status()))