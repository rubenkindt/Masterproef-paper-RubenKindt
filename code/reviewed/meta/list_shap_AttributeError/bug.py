from cpmpy import *

file ="C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/junk/3_jugs_regular1665052420275488"
m=Model().from_file(file)
nr = m.solve(solver="gurobi", time_limit=5*60)
print(m.status().exitstatus)
