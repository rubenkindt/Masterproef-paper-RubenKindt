from cpmpy import *

file = "exodus16650527724432807"
m = Model().from_file(file)
m.solve(solver="minizinc:set")
# m.solve(solver="minizinc:lcg")
# m.solve(solver="minizinc:int")
# m.solve(solver="minizinc:cp")
# m.solve(solver="minizinc:chuffed")