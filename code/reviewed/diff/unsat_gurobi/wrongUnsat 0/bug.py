from cpmpy import *

original = "hakank-bibd1665053540102282"
almostMinimized = "almostMinimized"

m4 = Model().from_file(almostMinimized)
m4.solve(solver="gurobi", time_limit=60*5)
print(m4)
print(str(m4.status()))

m2 = Model().from_file(almostMinimized)
m2.solve(time_limit=60*5)
print(str(m2.status()))
