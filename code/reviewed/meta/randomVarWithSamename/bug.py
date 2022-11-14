from cpmpy import *

# how the piclke was created
# if not os.path.exists("1"):
#     m = Model()
#     m += b == 4
#     m.to_file("1")

m2 = Model().from_file("1")
j = intvar(lb=3,ub=6)
m2 += j == 5
m2.solve() # solver independent
print(j.value())
print(m2.status().exitstatus)