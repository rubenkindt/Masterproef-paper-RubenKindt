from cpmpy import *

b  = intvar(lb=0,ub=2, name="sameName")
bb = intvar(lb=3,ub=4, name="sameName")
m = Model()
m += b == 0
m += bb == 3

m.solve(solver="minizinc:chuffed") # happens with all solvers
print(b.value())
print(bb.value())
print(m.status().exitstatus)