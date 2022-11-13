from cpmpy import *

file ="C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/junk/mario1665053194007364"
m=Model().from_file(file)

m.solve(solver='minizinc:gecode')
m.solve(solver='minizinc:gist')
m.solve(solver='minizinc:restart')
m.solve(solver='minizinc:xpress')
m.solve(solver='minizinc:set')
