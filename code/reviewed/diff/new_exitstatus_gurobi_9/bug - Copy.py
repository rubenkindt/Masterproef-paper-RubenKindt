from cpmpy import *

file = "minimized" # don't forget to unzip
m = Model().from_file(file)
m.solve(solver="ortools", time_limit=2) # timelimit is needed
print(m.status())