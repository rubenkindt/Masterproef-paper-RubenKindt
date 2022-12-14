import copy
import os
import re
import shutil
import argparse
import traceback
import random
import json
import minizinc

from cpmpy import *
from cpmpy.expressions.core import Comparison
from cpmpy.solvers.solver_interface import ExitStatus
from termcolor import colored


class metaModel():
    def __init__(self, solver=None, seedF=None):
        self.solver = solver
        self.seedFile = seedF
        self.metaRelations = []
        if self.seedFile is None:
            self.origModel = None
            self.modifModel = None
        else:
            try:
                self.origModel = Model().from_file(self.seedFile)
                self.modifModel = copy.deepcopy(self.origModel)
            except Exception as e:
                print(colored("failed to read seed" + str(e), "green", attrs=["bold"]))

def create_file(data, path):
    file = open(path, "w")
    file.write(data)
    file.close()

def recursiflySearch(cons, name):
    # returns a constraint with the matchin name, if found, else return None
    foundMax = None
    if not hasattr(cons, "args"):
        return foundMax
    for arg in cons.args:
        if hasattr(arg,"name"):
            if arg.name == name:
                foundMax = arg
                return foundMax
            else:
                foundMax = recursiflySearch(arg, name)
                if foundMax is not None:
                    return foundMax
    return foundMax

def getSeeds(path):
    seedPath = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.__contains__("."):
                continue
            seedPath.append((root, file))
    return seedPath

def recordCrash(mmodel, executionDir, seedFolder, seedName, solver, trace=None, errorName=None):
    # parts copied and or modified from STORM's soundness logging function

    # check if the crash folder exists
    if not os.path.exists(executionDir):
        os.mkdir(executionDir)
    path_to_crash_folder = os.path.join(executionDir, "crash")
    number_of_directories = 0
    for r, d, f in os.walk(path_to_crash_folder):
        number_of_directories = len(d)
        break

    if not os.path.exists(path_to_crash_folder):
        os.mkdir(path_to_crash_folder)

    # Create a directory for the crash
    safeErrorType = re.sub('[^a-zA-Z0-9 ]', '', errorName)  # remove all non (a-z A-Z 0-9 and " ") characters
    path_to_bug_dir = os.path.join(path_to_crash_folder, safeErrorType[:100] + str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)
    try:
        mmodel.modifModel.to_file(path_to_bug_dir + "/_Modif")
    except Exception:
        pass
    crash_logs = "seed: " + str(seedFolder + "/" + seedName) + "\n"
    crash_logs += "solver: " + str(mmodel.solver) + "\n"
    crash_logs += "metaRelations: " + str(mmodel.metaRelations) + "\n"
    crash_logs += "error reason: " + errorName + "\n"
    crash_logs += "error trace: " + trace + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "crash_logs.txt"))

def recordDiff(mmodel, executionDir, seedFolder, seedName):
    # parts copied and or modified from STORM's soundness logging function

    # check if the crash folder exists
    if not os.path.exists(executionDir):
        os.mkdir(executionDir)
    path_to_crash_folder = os.path.join(executionDir, "diff")
    number_of_directories = 0
    for r, d, f in os.walk(path_to_crash_folder):
        number_of_directories = len(d)
        break

    if not os.path.exists(path_to_crash_folder):
        os.mkdir(path_to_crash_folder)

    # Create a directory for the crash
    diff = str(mmodel.solver) + " " + str(mmodel.origModel.status().exitstatus.name) + " " + str(mmodel.modifModel.status().exitstatus.name)
    safeErrorType = re.sub('[^a-zA-Z0-9 ]', '', diff)  # remove all non safe
    path_to_bug_dir = os.path.join(path_to_crash_folder, safeErrorType[:100] + str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)
    try:
        mmodel.modifModel.to_file(path_to_bug_dir + "/_Modif")
    except Exception:
        pass
    crash_logs = "seed: " + str(seedFolder + "/" + seedName) + "\n" + "\n"
    crash_logs += "solver: " + str(mmodel.solver) + "\n"
    crash_logs += "modifications" + str(mmodel.metaRelations) + "\n"
    crash_logs += "difference: " + "original " + str(mmodel.origModel.status().exitstatus.name) + " VS " \
                  + "modified " + str(mmodel.modifModel.status().exitstatus.name) + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "diff_logs.txt"))

def semanticFusionIntInt(satModel, operation, invOperation):
    newcons = []
    firstCons = None
    secCons = None

    for i, cons in enumerate(satModel.modifModel.constraints):
        if firstCons is None and hasattr(cons, "args"):
            if (isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl)):
                firstCons = cons
                continue
        elif secCons is None and firstCons is not None and hasattr(cons, "args"):
            if (isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl)):
                secCons = cons
                continue
        newcons += [cons]

    if secCons is None:
        return newcons

    x = firstCons.args[0]
    y = secCons.args[0]
    # can be written with python build-in max and min, but those are overwritten by CPMpy,
    # writing this is easier then looking up how to use the overwritten build-ins
    maxLb = operation(x.lb, y.lb) if operation(x.lb, y.lb) > operation(x.ub, y.ub) else operation(x.ub, y.ub)
    minUb = operation(x.lb, y.lb) if operation(x.lb, y.lb) < operation(x.ub, y.ub) else operation(x.ub, y.ub)
    try:
        z = intvar(lb=maxLb, ub=minUb, name=("i" + str(random.randint(0,1000))))
    except Exception as e:
        return newcons
    xr = invOperation(z, y)
    yr = invOperation(z, x)
    firstCons.args[0] = xr
    secCons.args[0] = yr
    newcons += [(firstCons) & (secCons)]

    # replace some X, Y by xr, yr
    for j, cons in enumerate(newcons):
        if hasattr(cons, "args"):
            for i, arg in enumerate(cons.args):
                if hasattr(arg, "name") and arg.name == x.name and random.random() < 0.5:
                    cons.args[i] = xr
                if hasattr(arg, "name") and arg.name == y.name and random.random() < 0.5:
                    cons.args[i] = yr
    return newcons

def semanticFusionBoolBool(satModel, invOperation):
    newcons = []
    firstCons = None
    secCons = None

    if len(satModel.modifModel.constraints) <= 1:
        return satModel.modifModel.constraints

    randNr = [0, 0]
    while randNr[0] == randNr[1]:  # the randNr can be written better
        randNr = [random.randint(0, len(satModel.modifModel.constraints) - 1),
                  random.randint(0, len(satModel.modifModel.constraints) - 1)]

    for i, cons in enumerate(satModel.modifModel.constraints):
        if i == randNr[0]:
            firstCons = satModel.modifModel.constraints[i]
            continue
        if i == randNr[1]:
            secCons = satModel.modifModel.constraints[i]
            continue
        newcons += [cons]

    if secCons is None:
        return newcons

    x = firstCons
    y = secCons
    z = boolvar(name=("b"+str(random.randint(0,1000))))
    xr = invOperation(z, y)
    yr = invOperation(z, x)
    firstCons = xr
    secCons = yr
    newcons += [(firstCons) & (secCons)]

    # replace some X, Y by xr, yr
    for j, cons in enumerate(newcons):
        if hasattr(cons, "args"):
            for i, arg in enumerate(cons.args):
                if hasattr(arg, "name") and arg.name == x.name and random.random() < 0.5:
                    cons.args[i] = xr
                if hasattr(arg, "name") and arg.name == y.name and random.random() < 0.5:
                    cons.args[i] = yr
    return newcons

def satMutation(metaModel):
    metamorphicMutations = ["expandingAllDiff", "AllDiff~(==)", "expandAllEqual","AllEqual~(!=)", "addNewVar2AllEqual",
                            "TrueAndCons", "consAndCons2", "FalseOrCons", "xorCons", "==1",
                            "!=0", "==2>=|<=", ">=|<=2==", "sementicFusion+", "sementicFusion-",
                            "sementicFusion*", "sementicFusion^", "sementicFusion|", "sementicFusion&", "sementicFusion==",
                            "sementicFusion!=", "addRandomIntRestrictions", "True->cons", "cons->cons2", "cons1==cons2",
                            "addSmall2Max", "addSmall2Min", "addZero2Sum", "uselessAny", "uselessAll"]
    choice = random.choice(metamorphicMutations)
    metaModel.metaRelations += [choice]
    newcons = []
    if choice == "expandingAllDiff":
        for cons in metaModel.modifModel.constraints:
            if cons.name == 'alldifferent' and len(cons.args) <= 5:
                for i, arg1 in enumerate(cons.args):
                    for arg2 in cons.args[i+1:]:
                        newcons += [(arg1) != (arg2)]
            else:
                newcons += [cons]
    elif choice == "AllDiff~(==)":
        for cons in metaModel.modifModel.constraints:
            if cons.name == 'alldifferent' and len(cons.args) <= 5:
                for i, arg1 in enumerate(cons.args):
                    for arg2 in cons.args[i+1:]:
                        newcons += [~((arg1) == (arg2))]
            else:
                newcons += [cons]
    elif choice == "expandAllEqual":
        for cons in metaModel.modifModel.constraints:
            if cons.name == 'allequal' and len(cons.args) <= 5:
                for arg1 in cons.args[1:]:
                    newcons += [(cons.args[0]) == (arg1)]
            else:
                newcons += [cons]
    elif choice == "AllEqual~(!=)":
        for cons in metaModel.modifModel.constraints:
            if hasattr(cons, "name") and hasattr(cons, "args"):
                if cons.name == 'allequal' and len(cons.args) <= 5:
                    for i, arg1 in enumerate(cons.args):
                        for arg2 in cons.args[i+1:]:
                            newcons += [~((arg1) != (arg2))]
                else:
                    newcons += [cons]
            else:
                newcons += [cons]
    elif choice == "addNewVar2AllEqual":
        for cons in metaModel.modifModel.constraints:
            if cons.name == 'allequal':
                if isinstance(cons.args[0], variables._BoolVarImpl):
                    var = boolvar(name=("b"+str(random.randint(0,1000))))
                    allVar = cons.args+[var]
                    random.shuffle(allVar)
                    newcons += [AllEqual(allVar)]

                if isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl):
                    var = intvar(lb=cons.args[0].lb, ub=cons.args[0].ub, name=("i"+str(random.randint(0,1000))))
                    allVar = cons.args+[var]
                    random.shuffle(allVar)
                    newcons += [AllEqual(allVar)]
            else:
                newcons += [cons]
    elif choice == "TrueAndCons":
        for cons in metaModel.modifModel.constraints:
            if random.random() > 0.5:
                newcons += [(cons) & True]
            else:
                newcons += [True & (cons)]
    elif choice == "consAndCons2":
        for i, cons in enumerate(metaModel.modifModel.constraints):
            if len(metaModel.modifModel.constraints[i + 1:])>=1 and random.random() > 0.1:
                randCons = random.choice(metaModel.modifModel.constraints[i + 1:])
                newcons += [(randCons) & (cons) & (randCons)]
            else:
                newcons += [cons]
    elif choice == "FalseOrCons":
        for cons in metaModel.modifModel.constraints:
            if random.random() > 0.5:
                newcons += [False | (cons)]
            else:
                newcons += [(cons) | False]
    elif choice == "xorCons":
        for cons in metaModel.modifModel.constraints:
            newcons += [Xor([False, True, True] + [cons])]
    elif choice == "==1":
        for cons in metaModel.modifModel.constraints:
            newcons += [(cons) == 1]
    elif choice == "!=0":
        for cons in metaModel.modifModel.constraints:
            newcons += [(cons) != 0]
    elif choice == "==2>=|<=":
        for cons in metaModel.modifModel.constraints:
            if hasattr(cons, "name") and cons.name == "==":
                if random.random() < 0.5:
                    newcons += [(cons.args[0]) <= (cons.args[1])]
                else:
                    newcons += [(cons.args[0]) >= (cons.args[1])]
            else:
                newcons += [cons]
    elif choice == ">=|<=2==":
        for cons in metaModel.modifModel.constraints:
            if hasattr(cons, "name") and cons.name == "<=":
                newcons += [(cons.args[0]) < ((cons.args[1]) + 1)]
            elif hasattr(cons, "name") and cons.name == ">=":
                newcons += [((cons.args[0]) + 1) > (cons.args[1])]
            elif hasattr(cons, "name") and cons.name == "<":
                newcons += [((cons.args[0]) + 1) <= (cons.args[1])]
            elif hasattr(cons, "name") and cons.name == ">":
                newcons += [(cons.args[0]) >= ((cons.args[1]) + 1)]
            else:
                newcons += [cons]
    elif choice == "sementicFusion+":
        ope    = lambda x, y: x + y
        invOpe = lambda z, x: z - x
        newcons = semanticFusionIntInt(metaModel, operation=ope, invOperation=invOpe)
    elif choice == "sementicFusion-":
        ope    = lambda x, y: x + y
        invOpe = lambda z, x: z - x
        newcons = semanticFusionIntInt(metaModel, operation=ope, invOperation=invOpe)
    elif choice == "sementicFusion*":
        ope    = lambda x, y: x * y
        invOpe = lambda z, x: z / x
        newcons = semanticFusionIntInt(metaModel, operation=ope, invOperation=invOpe)
    elif choice == "sementicFusion^": # z=x^y inverse is x=z^y and y=z^x
        invOpe = lambda z, x: z ^ x
        newcons = semanticFusionBoolBool(metaModel, invOperation=invOpe)
    elif choice == "sementicFusion|": # z=x|y inverse is x=z|y and y=z|x
        invOpe = lambda z, x: z | x
        newcons = semanticFusionBoolBool(metaModel, invOperation=invOpe)
    elif choice == "sementicFusion&": # z=x&y inverse is x=z&y and y=z&x
        invOpe = lambda z, x: z & x
        newcons = semanticFusionBoolBool(metaModel, invOperation=invOpe)
    elif choice == "sementicFusion==": # z=x==y inverse is x=z==y and y=z==x
        invOpe = lambda z, x: z == x
        newcons = semanticFusionBoolBool(metaModel, invOperation=invOpe)
    elif choice == "sementicFusion!=": # z=x!=y inverse is x=z!=y and y=z!=x
        invOpe = lambda z, x: z != x
        newcons = semanticFusionBoolBool(metaModel, invOperation=invOpe)
    elif choice == "addRandomIntRestrictions":
        for loopVar in range(random.randint(1, 3)):
            i = intvar(lb=0, ub=10, shape=1, name="beep")
            if random.random() < 0.5:
                newcons += [i >= random.randint(1, 5)]
            else:
                newcons += [i <= random.randint(6, 9)]
            newcons += metaModel.modifModel.constraints
    elif choice == "True->cons":
        for cons in metaModel.modifModel.constraints:
            t = intvar(lb=0, ub=1, name=("i"+str(random.randint(0,1000))))
            temp = Model([t<=0]) #always True
            if random.random() < 0.2:
                newcons += [(temp.constraints[0]).implies(cons)]
            elif random.random() < 0.2:
                newcons += [(cons).implies(temp.constraints[0])]
            else:
                newcons += [(cons)]
    elif choice == "cons->cons2":
        for i, cons in enumerate(metaModel.modifModel.constraints):
            if random.random() < 0.2:
                cons2 = random.choice(metaModel.modifModel.constraints)
                newcons += [(cons)]
                newcons += [(cons).implies(cons2)]
            else:
                newcons += [(cons)]
    elif choice == "cons1==cons2":
        for i, cons in enumerate(metaModel.modifModel.constraints):
            if random.random() < 0.2:
                cons2 = random.choice(metaModel.modifModel.constraints)
                newcons += [(cons)]
                newcons += [(cons) == (cons2)]
            else:
                newcons += [(cons)]
    elif choice == "addSmall2Max":
        for cons in metaModel.modifModel.constraints:
            maxCons = recursiflySearch(cons, "max")
            if maxCons is not None:
                randArg = random.choice(maxCons.args)
                maxCons.args += (intvar(lb=randArg.lb, ub=randArg.ub+1, name=("i"+str(random.randint(0,1000)))),)
            newcons += [(cons)]
    elif choice == "addSmall2Min":
        for cons in metaModel.modifModel.constraints:
            minCons = recursiflySearch(cons, "min")
            if minCons is not None:
                randArg = random.choice(minCons.args)
                minCons.args += (intvar(lb=randArg.lb+1, ub=randArg.ub, name=("i"+str(random.randint(0,1000)))),)
            newcons += [cons]
    elif choice == "addZero2Sum":
        for cons in metaModel.modifModel.constraints:
            minCons = recursiflySearch(cons, "sum")
            if minCons is not None:
                minCons.args += [intvar(lb=0, ub=0, name=("i"+str(random.randint(0,1000))))]
            newcons += [cons]
    elif choice == "uselessAny":
        cons = intvar(lb=0, ub=1, name=("i"+str(random.randint(0,1000)))) == 0
        lst = [False, False, (cons)]
        random.shuffle(lst)
        newcons += [any(lst)]
        newcons += metaModel.modifModel.constraints
    elif choice == "uselessAll":
        cons = intvar(lb=0, ub=1, name=("i"+str(random.randint(0,1000)))) == 0
        lst = [True, (cons), True]
        random.shuffle(lst)
        newcons += [all(lst)]
        newcons += metaModel.modifModel.constraints
    else:
        NotImplementedError("this choice is not implemented: " + choice)

    random.shuffle(newcons)
    metaModel.modifModel = Model(newcons)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--startAt', type=int, nargs='?', help='the bug-catcher likes to fill the swap space and crash')
    parser.add_argument('--seed', type=int, nargs='?', help='seed for random')
    arguments = parser.parse_args()

    if arguments.startAt is None:
        arguments.startAt = 0
    if arguments.seed is None:
        arguments.seed = 123
    random.seed(arguments.seed)
    timeout = 2 * 60  # 5 minutes
    if os.name == 'posix':
        seedPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/metamorphic"
        resultsPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/metamorphic"
    else:
        seedPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/metamorphic"
        resultsPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/metamorphic"

    solvers = ['ortools', 'gurobi', 'pysat', 'pysat:cadical', 'pysat:gluecard3', 'pysat:gluecard4',
               'pysat:glucose3', 'pysat:glucose4', 'pysat:lingeling', 'pysat:maplechrono', 'pysat:maplecm',
               'pysat:maplesat', 'pysat:mergesat3', 'pysat:minicard', 'pysat:minisat22', 'pysat:minisat-gh',
               'minizinc:api', 'minizinc:cbc', 'minizinc:chuffed', 'minizinc:coin-bc', 'minizinc:coinbc',
               'minizinc:cp', 'minizinc:cplex', 'minizinc:experimental', 'minizinc:findmus', 'minizinc:float',
               'minizinc:gecode', 'minizinc:gist', 'minizinc:globalizer', 'minizinc:gurobi', 'minizinc:int',
               'minizinc:lcg', 'minizinc:mip', 'minizinc:ortools', 'minizinc:osicbc', 'minizinc:restart',
               'minizinc:scip', 'minizinc:set', 'minizinc:tool', 'minizinc:xpress']
    seedPaths = getSeeds(seedPath)
    random.shuffle(seedPaths)
    retry = 0
    for counter in range(len(seedPaths)):
        folder, fileName = seedPaths[counter]
        if counter < arguments.startAt:
            continue
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        mmodel = metaModel(solver=random.choice(solvers),seedF=folder + "/" + fileName)

        try:
            mmodel.origModel.solve(solver=mmodel.solver, time_limit=timeout)
            statusOri = mmodel.origModel.status().exitstatus.name
        except Exception as e:
            # print(colored("Crash of seed" + str(e), "white", attrs=["bold"]))
            if retry < 10:
                counter -= 1
                retry += 1
            else:
                retry = 0
            continue

        for i in range(1,5):
            satMutation(mmodel)

        try:
            mmodel.modifModel.solve(solver=mmodel.solver, time_limit=timeout)
            statusModi = mmodel.modifModel.status().exitstatus.name
        except json.decoder.JSONDecodeError as e:
            if str(e).__contains__("Expecting value: line 1 column"):
                continue
            else:
                print(colored("Crash" + str(e), "red", attrs=["bold"]))
                recordCrash(mmodel, executionDir=resultsPath, seedFolder=folder, seedName=fileName,
                            trace=traceback.format_exc(), errorName=str(e), solver=mmodel.solver)
                continue
        except minizinc.error.MiniZincError as e:  # all passed errors are already logged
            if str(e).__contains__("cannot load"):
                continue
            elif mmodel.solver == "minizinc:org.minizinc.mip.scip" and str(e).__contains__("Failed to load plugin"):
                continue
            elif mmodel.solver == "minizinc:org.minizinc.mip.xpress" and str(e).__contains__("Failed to load plugin"):
                continue
            elif mmodel.solver == "minizinc:xpress" and str(e).__contains__("Failed to load plugin"):
                continue
            elif mmodel.solver == "minizinc:scip" and str(e).__contains__("Failed to load plugin"):
                continue
            else:
                print(colored("Crash" + str(e), "red", attrs=["bold"]))
                recordCrash(mmodel, executionDir=resultsPath, seedFolder=folder, seedName=fileName,
                            trace=traceback.format_exc(), errorName=str(e), solver=mmodel.solver)
                continue
        except NotImplementedError as e:
            continue
        except Exception as e:
            print(colored("Crash" + str(e), "red", attrs=["bold"]))
            recordCrash(mmodel, executionDir=resultsPath, seedFolder=folder, seedName=fileName,
                        trace=traceback.format_exc(), errorName=str(e), solver=mmodel.solver)
            continue

        if statusOri == 'FEASIBLE' or statusOri == 'OPTIMAL':
            statusOri = "sat"
        if statusModi == 'FEASIBLE' or statusModi == 'OPTIMAL':
            statusModi = "sat"
        if statusOri != statusModi:
            print(colored("diff", "red", attrs=["bold"]))
            recordDiff(mmodel, executionDir=resultsPath, seedFolder=folder, seedName=fileName)
            continue


if __name__ == "__main__":
    __main__()