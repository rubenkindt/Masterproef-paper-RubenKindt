from cpmpy import *


file = "miss_manners16650519017740455"
file = "almostMinimized"

m = Model().from_file(file)
sol = m.solve(solver="ortools")
print(m)
print(m.status())
