from musx2 import *
from cpmpy import *

original = "wedding_optimal_chart16650535424371436"
minimized = "almostMinimized"

m3 = Model().from_file(minimized)
# m=mus(m3.constraints,solver="gurobi")
# m3=mus(m)
# m3.to_file("2")
m3.solve(solver="gurobi", time_limit=60*5)
print(str(m3.status()))

m2 = Model().from_file(minimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))