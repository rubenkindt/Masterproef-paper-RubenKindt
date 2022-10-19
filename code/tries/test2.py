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
# i=intvar(2,2,name="i")
# m=Model()
# m+= i==2
#
# m += ~(~(i==2))

file = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/soundness/0/minimized"

m = Model().from_file(file)
m.solve(solver="ortools")
m.status()
print(m.status())

m2 = Model().from_file(file)
m2.solve(solver="minizinc:chuffed")
print(m2.status())