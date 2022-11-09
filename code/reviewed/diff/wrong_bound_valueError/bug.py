from cpmpy import *

file ="3_coins1665053328231921"
m=Model().from_file(file)
m.solve(solver="pysat")
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