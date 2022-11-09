from cpmpy import *

file = "prob005_auto_correlation16650537061803274"

m2 = Model().from_file(file)
m2.solve(solver="ortools", time_limit=60*5)
print(m2.status())

m = Model().from_file(file)
p= m.solve(solver="gurobi", time_limit=60*5)
print(str(m.status()))