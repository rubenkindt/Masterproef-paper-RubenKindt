from simplificationWithError import *
from cpmpy import *

file ="MCDP116650517570588078"
m=Model().from_file(file)
m = simplification(m.constraints, solver="minizinc:experimental")
m = Model(m)
m.to_file("minimized")
#m.solve(solver="minizinc:chuffed", time_limit=30)
m.solve(solver='minizinc:ortools') # json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
m.solve(solver='minizinc:experimental') # json.decoder.JSONDecodeErrorJSONDecodeError: Expecting value: line 1 column 1 (char 0)
