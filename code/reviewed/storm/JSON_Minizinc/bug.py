from cpmpy import *file = "almostMinimized"m=Model().from_file(file)nr = m.solve(solver="minizinc:findmus", time_limit=5*60)