from cpmpy import *

file ="sudoku_17_hints16650532485163155"
file = "mutant_0"

m2 = Model().from_file(file)
print(m2.constraints)
m2.solve(solver="minizinc:chuffed")
print(m2.status())

m = Model().from_file(file)
m.solve(solver="ortools")
m.status()
print(m.status())


correctSet = m.constraints[5].args[0]
wrongSet = m.constraints[29].args[3]
print(correctSet.name + " = " + str(correctSet.value()), id(correctSet)) # grid[3,3] = 7
print(wrongSet.name + " = " + str(wrongSet.value()), id(wrongSet))     # grid[3,3] = 1