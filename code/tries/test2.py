from cpmpy import *
from cpmpy.solvers import *
from cpmpy.transformations.flatten_model import flatten_model

"""x = boolvar(shape=3)
m = Model()
m2 = Model()
m3 = Model()

m += (x[0] | x[1])
m += x[2]
m.to_file("test2_out")"""
m = Model.from_file("test2_out")


solver = CPM_minizinc(cpm_model=m, subsolver="chuffed")
solver.solve()
print(solver.user_vars)
print(solver.user_vars.pop().value())






#print(m.status())
