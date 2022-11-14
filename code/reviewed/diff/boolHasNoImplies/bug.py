from cpmpy import *

m = Model()
m += (1+1) > 3 # Could be a more complicated expression
m += (m.constraints[0]).implies(True)
m.solve(solver="minizinc:ortools")