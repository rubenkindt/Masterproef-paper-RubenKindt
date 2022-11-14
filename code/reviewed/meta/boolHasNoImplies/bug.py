from cpmpy import *

x = intvar(0, 4, shape=5, name="x")
m = Model()
m += 1==1 # find
m += (m.constraints[0]).implies(x[1]==2)
nr = m.solve() # solver independent