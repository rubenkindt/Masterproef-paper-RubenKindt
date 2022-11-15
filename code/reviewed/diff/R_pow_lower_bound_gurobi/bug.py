from cpmpy import *
from simplificationWithError import simplification

file = "prob005_auto_correlation16650537061803274"
# file = "minimized"

m = Model().from_file(file)
# i = intvar(lb=-10, ub=10)
# m += pow(i,2)==9
# m.solve(solver="gurobi")
# print(str(m.status()))


m = simplification(m.constraints, solver="gurobi")
m = Model(m)
m = m.to_file("2")