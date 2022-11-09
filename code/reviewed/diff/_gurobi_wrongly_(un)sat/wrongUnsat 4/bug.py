from cpmpy import *

original = "football1665051982904926"
minimized = "minimized"

m = Model().from_file(original)
m.solve(solver="gurobi", time_limit=60*5)
print(str(m.status()))

m2 = Model().from_file(minimized)
m2.solve(solver="gurobi", time_limit=60*5)
print(str(m2.status()))

m2 = Model().from_file(minimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))