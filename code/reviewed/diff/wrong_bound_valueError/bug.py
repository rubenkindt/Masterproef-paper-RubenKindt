from cpmpy import *

file ="3_coins1665053328231921"
file= "minimized"
m=Model().from_file(file)

b = boolvar(shape=3)
i = intvar(lb=0, ub=10)
m = Model()
m += sum([b[0], b[1], b[2]]) == i

m.solve(solver="ortools")
m.solve(solver="pysat:cadical")
m.solve(solver="pysat:minisat-gh")
m.solve(solver="pysat:minisat22")
m.solve(solver="pysat:minicard")
m.solve(solver="pysat:mergesat3")
m.solve(solver="pysat:maplesat")
m.solve(solver="pysat:maplecm")
m.solve(solver="pysat:maplechrono")
m.solve(solver="pysat:lingeling")
m.solve(solver="pysat:glucose4")
m.solve(solver="pysat:glucose3")
m.solve(solver="pysat:gluecard3")
m.solve(solver="pysat:gluecard4")