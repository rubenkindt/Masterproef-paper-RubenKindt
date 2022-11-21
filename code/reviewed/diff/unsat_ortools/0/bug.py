from mus import *
from cpmpy import *
from simplification import *


file = "bibd16650522531582546"
file = "almostMinimal"

m = Model().from_file(file)
sol = m.solve(solver="minizinc:ortools")
print(m.status())
# print(m)
# m = mus(soft=m.constraints,solver="gurobi") # m = simplification(soft=m.constraints,solver="gurobi")
# m = mus(m)
# m.to_file("2")

m = Model().from_file(file)
sol = m.solve(solver="ortools")
print(m.status())