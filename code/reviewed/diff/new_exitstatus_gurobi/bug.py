from simplificationWithError import *
from cpmpy import *

file = "steiner16650517652597234"
file = "minimized"
solver = "gurobi"

m = Model().from_file(file)
# m = simplification(m.constraints)
# m= Model(m)
# m.to_file("minimized")
m.solve(solver=solver, time_limit=5)
print(m.status())