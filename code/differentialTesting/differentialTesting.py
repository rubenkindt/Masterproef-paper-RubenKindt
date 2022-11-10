import os
import re
import shutil
import argparse
import traceback
import random
import json
import minizinc

from cpmpy import *
from gurobipy import GurobiError
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

def recordCrash(executionDir, seedFolder, seedName, solver, solveAll, trace=None, errorName=None):
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

    crash_logs = "solveAll" if solveAll else "solve"
    crash_logs += "\n"
    crash_logs += "seed: " + str(seedFolder + "/" + seedName) + "\n"
    crash_logs += "solver: " + str(solver) + "\n"
    crash_logs += "error reason: " + errorName + "\n"
    crash_logs += "error trace: " + trace + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "crash_logs.txt"))


def recordDiff(executionDir, seedFolder, seedName, solveAll, potentialNrDiff=None, potentialStatusDiff=None):
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
    statusDiffString = str(potentialStatusDiff)
    safeErrorType = re.sub('[^a-zA-Z0-9 ]', '', statusDiffString)  # remove all non (a-z A-Z 0-9 and " ") characters
    path_to_bug_dir = os.path.join(path_to_crash_folder, safeErrorType[:100] + str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)

    crash_logs = "solveAll" if solveAll else "solve"
    crash_logs += "\n"
    crash_logs += "seed: " + str(seedFolder + "/" + seedName) + "\n" + "\n"
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
    parser.add_argument('--seed', type=int, nargs='?',
                        help='seed')
    args = parser.parse_args()
    if os.name == 'posix':
        seedPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
    else:
        seedPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
    seedPaths = getSeeds(seedPath)
    random.seed(args.seed)
    random.shuffle(seedPaths)

    for counter in range(len(seedPaths)):
        folder, fileName = seedPaths[counter]
        if counter < args.startAt:
            continue
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        differentialTest(folder, fileName, args.seed)

def differentialTest(folder, fileName, seed=123):
    solvers = ['ortools', 'gurobi', 'pysat', 'pysat:cadical', 'pysat:gluecard3', 'pysat:gluecard4',
               'pysat:glucose3', 'pysat:glucose4', 'pysat:lingeling', 'pysat:maplechrono', 'pysat:maplecm',
               'pysat:maplesat', 'pysat:mergesat3', 'pysat:minicard', 'pysat:minisat22', 'pysat:minisat-gh',
               'minizinc:api', 'minizinc:cbc', 'minizinc:chuffed', 'minizinc:coin-bc', 'minizinc:coinbc',
               'minizinc:cp', 'minizinc:cplex', 'minizinc:experimental', 'minizinc:findmus', 'minizinc:float',
               'minizinc:gecode', 'minizinc:gist', 'minizinc:globalizer', 'minizinc:gurobi', 'minizinc:int',
               'minizinc:lcg', 'minizinc:mip', 'minizinc:ortools', 'minizinc:osicbc', 'minizinc:restart',
               'minizinc:scip', 'minizinc:set', 'minizinc:tool', 'minizinc:xpress']
    timeout = 1 * 60
    if os.name == 'posix':
        executionPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"
    else:
        executionPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"

    for solveAll in [True, False]:
        nrOfsol = []
        status = []
        for solver in solvers:
            try:
                m = Model().from_file(folder + "/" + fileName)
                if solveAll:
                    sol = m.solveAll(solver=solver, time_limit=timeout, solution_limit=100)
                    s = m.status().exitstatus.name
                    nrOfsol.append((sol, solver))
                else:
                    m.solve(solver=solver, time_limit=timeout)
                    s = m.status().exitstatus.name
                if s == "OPTIMAL" or s == "FEASIBLE":
                    s = "sat"
                status.append((s, solver))

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
                                trace=traceback.format_exc(), errorName=str(e), solver=solver, solveAll=solveAll)
            except GurobiError as e:
                if str(e).__contains__("variable is less than zero for POW function"):
                    pass
                else:
                    raise e
            except json.decoder.JSONDecodeError as e:
                if str(e).__contains__("Expecting value: line 1 column"):
                    pass
                else:
                    print(colored("Crash " + str(e), "red", attrs=["bold"]))
                    recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                trace=traceback.format_exc(), errorName=str(e), solver=solver, solveAll=solveAll)
            except ValueError as e:
                if solver == "pysat:minisat-gh" and str(e).__contains__("Wrong bound: "):
                    pass
                elif solver == "pysat:minisat22" and str(e).__contains__("Wrong bound: "):
                    pass
                elif solver == "pysat:minicard" and str(e).__contains__("Wrong bound: "):
                    pass
                elif solver == "pysat:mergesat3" and str(e).__contains__("Wrong bound: "):
                    pass
                elif solver == "pysat:maplesat" and str(e).__contains__("Wrong bound: "):
                    pass
                elif solver == "pysat:maplecm" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:maplechrono" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:lingeling" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:glucose4" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:glucose3" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:gluecard3" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:gluecard4" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat:cadical" and str(e).__contains__( "Wrong bound: "):
                    pass
                elif solver == "pysat" and str(e).__contains__("Wrong bound:"):
                    pass
                else:
                    print(colored("Crash " + str(e), "red", attrs=["bold"]))
                    recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                trace=traceback.format_exc(), errorName=str(e), solver=solver, solveAll=solveAll)
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
                                trace=traceback.format_exc(), errorName=str(e), solver=solver, solveAll=solveAll)
            except NotImplementedError as e:
                pass
            except RuntimeError as e:
                if str(e) == "Event loop is closed":
                    pass
                else:
                    raise e
            except Exception as e:
                if str(e) == "CPM_pysat: only satisfaction, does not support an objective function":
                    pass
                elif str(e).__contains__("pickle.UnpicklingError"):
                    pass # did not read in a pickle file
                else:
                    # crash
                    print(colored("Crash " + str(e), "red", attrs=["bold"]))
                    recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                                trace=traceback.format_exc(), errorName=str(e), solver=solver, solveAll=solveAll)

        if len(status) == 0:
            continue
        if solveAll:
            if len(nrOfsol) == 0:
                continue
            nrs=[]
            for i in nrOfsol:
                nrs.append(i[0])
            if len(set(nrs)) <= 1:# and len(set(status)) <= 1:
                continue
            recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status, solveAll=solveAll)
        else:
            statuses=[]
            for i in status:
                statuses.append(i[0])
            if len(set(statuses)) <= 1:
                continue
            recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status, solveAll=solveAll)

if __name__ == "__main__":
    __main__()
