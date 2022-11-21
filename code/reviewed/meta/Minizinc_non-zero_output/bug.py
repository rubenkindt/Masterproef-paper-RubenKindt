from cpmpy import *

file ="almostMinimal"
m=Model().from_file(file)

m.solve(solver='minizinc:gecode')
m.solve(solver='minizinc:gist')
m.solve(solver='minizinc:restart')
m.solve(solver='minizinc:xpress')
m.solve(solver='minizinc:set')
