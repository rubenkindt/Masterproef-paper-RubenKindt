from simplificationWithError import *
from cpmpy import *

file ="prob050_diamond_free1665053790857216"
m=Model().from_file(file)
m.solve(solver='minizinc:experimental')
m.solve(solver='minizinc:osicbc')
