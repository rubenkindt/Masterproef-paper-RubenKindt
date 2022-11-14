from simplificationWithError import *
from cpmpy import *

file ="C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/junk/mario1665053194007364"
file = "minimized"
m=Model().from_file(file)

m.solve(solver='minizinc:xpress')
