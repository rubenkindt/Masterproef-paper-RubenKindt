import os
import re
import shutil
import argparse
import traceback
import random
import json
import minizinc

from cpmpy import *
from termcolor import colored


def create_file(data, path): # copied from STORM (https://github.com/Practical-Formal-Methods/storm)
    file = open(path, "w")
    file.write(data)
    file.close()

def getSeeds(path):
    seedPath = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.__contains__("."):
                continue
            seedPath.append((root, file))
    return seedPath

def recordCrash(executionDir, seedFolder, seedName, solver, trace=None, errorName=None):
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
    path_to_bug_dir = os.path.join(path_to_crash_folder, safeErrorType + str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)

    crash_logs = "seed: " + str(seedFolder + "/" + seedName) + "\n"
    crash_logs += "solver: " + str(solver) + "\n"
    crash_logs += "error reason: " + errorName + "\n"
    crash_logs += "error trace: " + trace + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "crash_logs.txt"))


def recordDiff(executionDir, seedFolder, seedName, potentialNrDiff=None, potentialStatusDiff=None):
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
    path_to_bug_dir = os.path.join(path_to_crash_folder, str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)

    crash_logs = "seed: " + str(seedFolder + "/" + seedName) + "\n" + "\n"
    if len(potentialNrDiff) > 1:
        crash_logs += "difference in amount of solution: "
        crash_logs += str(potentialNrDiff) + "\n"
    if len(potentialStatusDiff) > 1:
        crash_logs += "difference in status: "
        crash_logs += str(potentialStatusDiff) + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "diff_logs.txt"))

def __main__():
    parser = argparse.ArgumentParser(description='--startAt for starting at a specify seed nr')
    parser.add_argument('--startAt', type=int, nargs='?',
                        help='the bug-catcher likes to fill the swap space and crash ')
    args = parser.parse_args()
    solveAll = False
    if solveAll:
        solvers = ["gurobi", "ortools"] # "minizinc:gurobi", , "minizinc:chuffed"]
    else:
        
        solvers = ['ortools', 'gurobi', 'pysat', 'pysat:cadical', 'pysat:gluecard3', 'pysat:gluecard4',
                   'pysat:glucose3', 'pysat:glucose4', 'pysat:lingeling', 'pysat:maplechrono', 'pysat:maplecm',
                   'pysat:maplesat', 'pysat:mergesat3', 'pysat:minicard', 'pysat:minisat22', 'pysat:minisat-gh',
                    'minizinc:api', 'minizinc:cbc', 'minizinc:chuffed', 'minizinc:coin-bc', 'minizinc:coinbc',
                    'minizinc:cp', 'minizinc:cplex', 'minizinc:experimental', 'minizinc:findmus', 'minizinc:float',
                    'minizinc:gecode', 'minizinc:gist', 'minizinc:globalizer', 'minizinc:gurobi', 'minizinc:int',
                    'minizinc:lcg', 'minizinc:mip', 'minizinc:org.chuffed.chuffed', 'minizinc:org.gecode.gecode',
                    'minizinc:org.gecode.gist', 'minizinc:org.minizinc.findmus', 'minizinc:org.minizinc.globalizer',
                    'minizinc:org.minizinc.mip.coin-bc', 'minizinc:org.minizinc.mip.cplex',
                    'minizinc:org.minizinc.mip.gurobi', 'minizinc:org.minizinc.mip.scip',
                    'minizinc:org.minizinc.mip.xpress', 'minizinc:osicbc', 'minizinc:restart', 'minizinc:scip',
                    'minizinc:set', 'minizinc:tool', 'minizinc:xpress']
    timeout = 5 * 60  # 5 minutes
    if os.name == 'posix':
        seedPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
        executionPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"
    else:
        seedPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
        executionPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"

    seedPaths = getSeeds(seedPath)
    random.seed(123)
    random.shuffle(seedPaths)
    counter = 0
    for folder, fileName in seedPaths:
        counter += 1
        if counter < args.startAt:
            continue
        nrOfsol = []
        status = []
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        for solver in solvers:
            if solveAll:
                try:
                    m = Model().from_file(folder + "/" + fileName)
                    sol = m.solveAll(solver=solver, time_limit=timeout, solution_limit=100)
                    nrOfsol.append((sol, solver))
                    status.append((m.status().exitstatus.name, solver))
                except Exception as e:
                    # crash
                    print(colored("Crash " + str(e), "red", attrs=["bold"]))
                    recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                trace=traceback.format_exc(), errorName=str(e), solver=solver)
            else:
                try:
                    m = Model().from_file(folder + "/" + fileName)
                    m.solve(solver=solver, time_limit=timeout)
                    s = m.status().exitstatus.name
                    if s == "OPTIMAL" or s == "FEASIBLE":
                        s = "sat"
                    status.append((s, solver))
                except NotImplementedError as e:
                    pass
                except minizinc.error.MiniZincError as e: # all passed errors are already logged
                    if str(e).__contains__("cannot load"):
                        pass
                    elif solver == "minizinc:gecode" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:gist" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:org.gecode.gecode" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:org.gecode.gist" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:restart" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:set" and str(
                            e) == "MiniZinc stopped with a non-zero exit code, but did not output an error message. ":
                        pass
                    elif solver == "minizinc:org.minizinc.mip.scip" and str(e).__contains__("Failed to load plugin"):
                        pass
                    elif solver == "minizinc:org.minizinc.mip.xpress" and str(e).__contains__("Failed to load plugin"):
                        pass
                    elif solver == "minizinc:xpress" and str(e).__contains__("Failed to load plugin"):
                        pass
                    elif solver == "minizinc:scip" and str(e).__contains__("Failed to load plugin"):
                        pass
                    else:
                        print(colored("Crash " + str(e), "red", attrs=["bold"]))
                        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                    trace=traceback.format_exc(), errorName=str(e), solver=solver)
                except json.decoder.JSONDecodeError as e:
                    if str(e).__contains__("Expecting value: line 1 column"):
                        pass
                    else:
                        print(colored("Crash " + str(e), "red", attrs=["bold"]))
                        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                    trace=traceback.format_exc(), errorName=str(e), solver=solver)
                except ValueError as e:
                    if solver == "pysat:minisat-gh" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:minisat22" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:minicard" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:mergesat3" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:maplesat" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:maplecm" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:maplechrono" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:lingeling" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:glucose4" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:glucose3" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:gluecard3" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:gluecard4" and str(e) == "Wrong bound: last_val":
                        pass
                    elif solver == "pysat:cadical" and str(e) == "Wrong bound: last_val":
                        pass
                    else:
                        print(colored("Crash " + str(e), "red", attrs=["bold"]))
                        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                    trace=traceback.format_exc(), errorName=str(e), solver=solver)
                except AttributeError as e:
                    if solver == "gurobi" and str(e) == "'list' object has no attribute 'shape'":
                        pass
                    elif solver == "gurobi" and str(e) == "'bool' object has no attribute 'is_bool'":
                        pass
                    elif solver == "ortools" and str(e) == "'bool' object has no attribute 'is_bool'":
                        pass
                    else:
                        print(colored("Crash " + str(e), "red", attrs=["bold"]))
                        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                    trace=traceback.format_exc(), errorName=str(e), solver=solver)
                except Exception as e:
                    if str(e) == "CPM_pysat: only satisfaction, does not support an objective function":
                        pass
                    else:
                        # crash
                        print(colored("Crash " + str(e), "red", attrs=["bold"]))
                        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                    trace=traceback.format_exc(), errorName=str(e), solver=solver)

        if solveAll:
            if len(status) == 0 or len(nrOfsol) == 0:
                continue
            nrs=[]
            for i in nrOfsol:
                nrs.append(i[0])
            if len(set(nrs)) <= 1:# and len(set(status)) <= 1:
                continue
            recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status)
        else:
            if len(status) == 0:
                continue
            statuses=[]
            for i in status:
                statuses.append(i[0])
            if len(set(statuses)) <= 1:
                continue
            recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status)

if __name__ == "__main__":
    __main__()
