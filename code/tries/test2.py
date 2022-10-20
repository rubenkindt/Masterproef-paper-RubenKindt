from cpmpy import *
from cpmpy.transformations.flatten_model import *
from musx import musx

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

file ="C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/try2/error/1/mutant_221"

m2 = Model().from_file(file)
#mu = musx(m2.constraints)
#mu.to_file("C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/try2/error/1/minimized")
m2.solve()
print(m2.status())

m = Model().from_file(file)
m.solve(solver="ortools")
m.status()
print(m.status())