from cpmpy import *

# how the piclke was created
# if not os.path.exists("1"):
#     m = Model()
#     m += b == 4
#     m.to_file("1")

b = intvar(lb=3,ub=6)
m2 = Model().from_file("1")
m2 += b == 5
m2.solve(solver="minizinc:ortools") # happens with all solvers
print(b.value())
print(m2.status().exitstatus)