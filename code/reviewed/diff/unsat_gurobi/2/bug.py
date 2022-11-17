from musx2 import *
from cpmpy import *

original = "football1665051982904926"
minimized = "almostMinimized"

m2 = Model().from_file(minimized)

# m=mus(m2.constraints,solver="gurobi")
# m2=mus(m)
# m2.to_file("2")
m2.solve(solver="gurobi", time_limit=60*5)
print(m2)
print(str(m2.status()))

m2 = Model().from_file(minimized)
m2.solve(solver="ortools", time_limit=60*5)
print(str(m2.status()))