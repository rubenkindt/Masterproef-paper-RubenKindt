from cpmpy import *

minimizedFile = "minimized"
m = Model().from_file(minimizedFile)
m.solve(solver="ortools")
print(m.status())
print(m)
correctSet = m.constraints[0].args[0]
wrongSet = m.constraints[1].args[0].args[0].args[0]
print(correctSet.name + " = " + str(correctSet.value()))
print(wrongSet.name + " = " + str(wrongSet.value()))


m2 = Model()
i = intvar(lb=0, ub=9, shape=(4,4), name="grid")
m2 += i[3,3] == 7
m2 += ~~(i[3,3] == 7)
m2.solve(solver="ortools")
print(m2.status())
print(m2)
correctSet = m2.constraints[0].args[0]
wrongSet = m2.constraints[1].args[0].args[0].args[0]
print(correctSet.name + " = " + str(correctSet.value()))
print(wrongSet.name + " = " + str(wrongSet.value()))
