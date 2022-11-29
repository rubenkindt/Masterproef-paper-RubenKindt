from cpmpy import *

file = "minimized"
m=Model().from_file(file)

from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
from cpmpy.transformations.get_variables import print_variables
mf = flatten_model(m)
print_variables(mf)
print(mf)
print(m)

m.solve(solver='minizinc:gecode')
m.solve(solver='minizinc:gist')
m.solve(solver='minizinc:restart')
m.solve(solver='minizinc:xpress')
m.solve(solver='minizinc:set')
