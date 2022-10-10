from cpmpy import *
from cpmpy.solvers import *
from cpmpy.transformations.flatten_model import flatten_model

x = boolvar(shape=3)
m = Model()
m2 = Model()
m3 = Model()

t2 = (x[0] & x[1])
t  = t2 ^ x[2]
s =  ~(t)
m += t
print(m.constraints)

#m.to_file("test2_out")
#m.to_file(fname = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/test3_out")

solver = CPM_minizinc(cpm_model=m, subsolver="chuffed")
solver.solve(time_limit=120)
print(solver.status())
#print(t.value())
#print(t2.value())
#print(s.value())
#print(x.value())
