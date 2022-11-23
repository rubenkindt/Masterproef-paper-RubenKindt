from mus import *
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables, get_variables_model

file = "_Modif"
file = "almostedMinimized"
solver = "gurobi"

m = Model().from_file(file) # randomize constraints and it becomes sat
# m = Model(m.constraints)
# m = mus_naive(m.constraints, solver=solver)
# m = Model(m)
# m.to_file("2")

m.solve(solver=solver)
print(m.status())

m2 = Model().from_file(file)
m2.solve(solver="ortools")#, solution_limit=100)
print(m2.status())

# print(m2)


# for i in get_variables_model(m2):
#     print(i + ".lb="+str(i.lb) +",ub="+str(i.ub)+"name="+str(i.name))