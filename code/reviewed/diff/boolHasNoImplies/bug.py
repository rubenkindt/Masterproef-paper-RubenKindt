from cpmpy import *

b = boolvar()
m = Model()
m += b==1
m += (m.constraints[0]).implies(True)
nr = m.solve()