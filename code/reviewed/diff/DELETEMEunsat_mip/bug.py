from musx2 import *
from simplification import simplification
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables, get_variables_model

file = "exodus16650527724432807"
# file = "minimized"

m = Model().from_file(file)
# m=mus(m.constraints,solver="gurobi")
# m=mus(m)
# m.to_file("2")
m.solve(solver="minizinc:mip") # same with osicbc, float, coinbc


m = Model().from_file(file)
s = SolverLookup.get("minizinc:chuffed", m)
print("".join(map(str, s.mzn_model._code_fragments)))
# s.solve()
print(s.status())

for i in get_variables_model(m):
    print(i + ".lb="+str(i.lb) +",ub="+str(i.ub)+"name="+str(i.name))