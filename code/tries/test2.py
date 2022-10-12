from cpmpy import *
from cpmpy.solvers import *
from cpmpy.transformations.flatten_model import flatten_model


# x = boolvar(shape=3)
# m = Model()
# m2 = Model()
# m3 = Model()
#
# t = (x[0]) & (x[1])
# # t  = t2 ^ x[2]
# # s =  ~(t)
# m += t
# print(m.constraints)
i=intvar(0,100000000,3)


b = boolvar(shape=3)
cons = ~(b[0] | b[1])
m = Model([(cons)])
m += ~i[0]<1
print(m.constraints)
#m = Model().from_file("C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/flattened/coins_grid166505189604997")

m.solve(solver="ortools")
#print(s.status())
print(m.status())
print(m.constraints)
print(i.value())
print(b[0].value())
print(b[1].value())



#cons_val = cons.value()
#m.to_file("test2_out")
#model = Model().from_file(fname = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/tries/mutant_0")
#model = Model().from_file(fname = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/flattened/coins_grid166505189604997")
# print(model.constraints)
# model2 = flatten_model(model)
# print(model2.constraints)


# SolverLookup.get("minizinc",model).solve(time_limit=120)
# print(model.status())


#print(t.value())
#print(t2.value())
#print(s.value())
#print(x.value())