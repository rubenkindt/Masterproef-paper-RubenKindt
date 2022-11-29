from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

b = boolvar(name="b")
i = intvar(lb=0, ub=1, name="i")
k = intvar(lb=0, ub=1, name="k")

m = Model()
m += (b == False) & (b != (k == 1))
m += (i == 0) == (b != (k == 1))
m += i == 0

# m = flatten_model(m)

m.solve(solver="minizinc:gurobi")
print(m.status())
m.solve(solver="gurobi")
print(m.status())
m.solve(solver="minizinc:ortools")
print(m.status())
m.solve(solver="ortools")
print(m.status())


from cpmpy.transformations.get_variables import get_variables, get_variables_model
for i in get_variables_model(m):
    print(i + ".lb="+str(i.lb) +",ub="+str(i.ub)+"name="+str(i.name)+" value "+str(i._value))