import random
import math
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

def simplification(constraints, solver="gurobi", verbose=False):
    #isolation technique

    m = Model(constraints)
    #m = flatten_model(m)
    m.solve(solver=solver)
    status = m.status().exitstatus.name
    if status != "UNSATISFIABLE":
        return []

    failing = m.constraints
    granularity = 2
    shuffled = 0
    while True:
        print("failing length: " + str(len(failing)) + " shuffled:" + str(shuffled))
        for i in range(granularity):
            lb = math.floor(len(failing) / granularity) * i
            ub = math.ceil(len(failing) / granularity) * (i + 1)
            newPart = failing[lb:ub]
            m = Model(newPart)
            
            if len(newPart) <= 1:
                return failing
            m.solve(solver=solver)
        
            status = m.status().exitstatus.name
            if status == "UNSATISFIABLE":
                failing = m.constraints
                granularity = 2
                shuffled = 0
                break
            if i >= granularity - 1:
                granularity += 1
                if granularity >= 10:
                    if shuffled >= 2:
                        return failing
                    shuffled += 1
                    random.shuffle(failing)
                    granularity = 2