from cpmpy import *

original = "hakank-bibd1665053540102282"
minimized = "minimized"

m4 = Model().from_file("minimized")
m4.solve(solver="gurobi", time_limit=60*5)
print(str(m4.status()))

m2 = Model().from_file("minimized")
m2.solve(time_limit=60*5)
print(str(m2.status()))