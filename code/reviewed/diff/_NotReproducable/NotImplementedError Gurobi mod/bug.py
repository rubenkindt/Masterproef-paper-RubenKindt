from cpmpy import *

file = "hidato_table16650535508907185"

m = Model()
p= m.solveAll(solver="gurobi", time_limit=2*60, solution_limit=100)
print(str(m.status()))

