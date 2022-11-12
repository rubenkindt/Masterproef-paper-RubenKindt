from cpmpy import *

original = "bibd16650522531619968"
minimized = "minimized"

m2 = Model().from_file(minimized)
m2.solve(solver="gurobi", time_limit=60*5)
print(str(m2.status()))

m2 = Model().from_file(minimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))