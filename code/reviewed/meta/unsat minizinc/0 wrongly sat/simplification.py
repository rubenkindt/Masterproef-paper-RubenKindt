import random
import math
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

def simplification(constraints, solver="gurobi", verbose=False):
    #isolation technique

    m = Model(constraints)
    m = flatten_model(m)
    m.solve(solver=solver)
    status = m.status().exitstatus.name
    if status != "UNSATISFIABLE":
        raise Exception("not unsat")

    failing = m.constraints
    granularity = 2
    shuffled = 0
    counter = 0
    while True:
        print("failing length: " + str(len(failing)) + " shuffled:" + str(shuffled))
        for i in range(granularity):
            lb = math.floor(len(failing) / granularity) * i
            ub = math.ceil(len(failing) / granularity) * (i + 1)
            newPart = failing[lb:ub]
            m = Model(newPart)
            
            m.solve(solver=solver)
        
            status = m.status().exitstatus.name
            if status == "UNSATISFIABLE":
                failing = m.constraints
                granularity = 2
                shuffled = 0
                if len(newPart) <= 1:
                    return failing
                break
            if i >= granularity - 1:
                granularity += 1
                counter += 1
                if counter > 20:
                    return failing
                if granularity >= 4:
                    if shuffled >= 5:
                        return failing
                    shuffled += 1
                    random.shuffle(failing)
                    granularity = 2