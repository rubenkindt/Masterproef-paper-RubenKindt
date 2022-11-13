from cpmpy import *

b = boolvar()
m = Model()
m += True
m += (m.constraints[0]).implies(True)
nr = m.solve()
print(m.status())