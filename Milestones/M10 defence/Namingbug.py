from cpmpy import *

i = intvar(lb=0, ub=9, name="+i")
m = Model()
m += i > 0

m.solve(solver="minizinc:chuffed") # Crash

s = SolverLookup.get("minizinc:chuffed", m)
print("".join(map(str, s.mzn_model._code_fragments)))

