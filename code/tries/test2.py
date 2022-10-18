from cpmpy import *
from cpmpy.transformations.flatten_model import *
from musx import musx

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
# i=intvar(0,100000000,3)
#
#
# b = boolvar(shape=3)
# cons = ~(b[0] | b[1])
# m = Model([(cons)])
# m += ~i[0]<1
# str=m.constraints[0].name
# print(m.constraints[0].name)
i=intvar(2,2,name="i")
m=Model()
m+= i==2

m += ~(~(i==2))

#m = Model().from_file("C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/soundness/0/mutant_0")


#m=flatten_model(m)
m.solve(solver="minizinc:osicbc")
m.status()
print(m.status())

# unsat_cons = musx(m.constraints)
# model2 = Model(unsat_cons)
# model2.solve()
# print(model2.status())

#
# res = m.solve(solver="ortools")
# print(res)


#print(s.status())
# print(m.status())
# print(b[0].value())
# print(m.constraints)
# print(i.value())
# print(b[1].value())

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