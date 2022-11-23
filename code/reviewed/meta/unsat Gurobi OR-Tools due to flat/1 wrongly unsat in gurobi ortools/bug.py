from mus import *
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables, get_variables_model

file = "_Modif"
file = "minimal"
solver = "minizinc:gurobi"

m = Model().from_file(file)
m.solve(solver=solver)
print(m.status())

m.solve(solver="gurobi")
print(m.status())
m.solve(solver="minizinc:ortools")
print(m.status())
m.solve(solver="ortools")
print(m.status())

print(m)

# m = Model().from_file(file)
# s = SolverLookup.get("minizinc:chuffed", m)
# print("".join(map(str, s.mzn_model._code_fragments)))
# s.solve()
# print(s.status())

for i in get_variables_model(m):
    print(i + ".lb="+str(i.lb) +",ub="+str(i.ub)+"name="+str(i.name)+" value "+str(i._value))