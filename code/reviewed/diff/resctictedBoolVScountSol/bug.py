from cpmpy import *

b = boolvar(shape=3, name="b")
m = Model()
# m += b[0]==1
# m += ~(b[0] | b[1])
# #m += (b[0] + b[1]) == 1

nr = m.solveAll(solver="solver:chuffed", display=b)

print(nr)
print(m.status())
