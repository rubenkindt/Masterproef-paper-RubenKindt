from cpmpy import *

original = "wedding_optimal_chart16650535424371436"
minimized = "almostMinimized"

m3 = Model().from_file(minimized)
m3.solve(solver="gurobi", time_limit=60*5)
print(str(m3.status()))

m2 = Model().from_file(minimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))