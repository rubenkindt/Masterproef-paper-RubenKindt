from cpmpy import *

original = "miss_manners16650518983839905"
minimized = "almostMinimized"

m2 = Model().from_file(minimized)
m2.solve(solver="gurobi", time_limit=60*5)
print(m2)
print(str(m2.status()))

m2 = Model().from_file(minimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))