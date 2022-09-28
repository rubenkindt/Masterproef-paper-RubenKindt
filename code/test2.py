from cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")
s += sum(x) <= 5
# we are operating on a ortools' interface here
s.solution_hint(x, [1,2,3])
Model.solve()
print(x.value())