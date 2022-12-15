from cpmpy import *
from cpmpy.transformations.get_variables import print_variables

i = intvar(lb=3, ub=6, name="+int")
m = Model()
m += i > 4

s = SolverLookup.get("minizinc", m)
print("".join(map(str, s.mzn_model._code_fragments)))

print_variables(m)
print(m)
m.solve(solver="minizinc:chuffed") # Crash

