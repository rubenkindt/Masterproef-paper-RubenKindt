from cpmpy import *

original = "football1665051982904926"
minimized = "minimized"

m2 = Model().from_file(minimized)
m2.solve(solver="gurobi", time_limit=60*5)
print(str(m2.status()))

m2 = Model().from_file(minimized)
m2.solve(solver="ortools", time_limit=60*5)
print(str(m2.status()))