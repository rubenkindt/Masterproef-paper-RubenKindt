from cpmpy import *

x = intvar(0, 4, shape=5, name="x")
m = Model()
m += 1==1
m += (m.constraints[0]).implies(x[1]==2)
nr = m.solve(solver="gurobi", time_limit=5*60)