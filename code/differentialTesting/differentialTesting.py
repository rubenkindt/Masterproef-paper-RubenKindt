import os
import re
import shutil
import argparse
import gc
import traceback

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


    solvers = ["ortools", "gurobi"] # "minizinc:gurobi", , "minizinc:chuffed"]
    timeout = 5 * 60  # 5 minutes
    if os.name == 'posix':
        seedPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
        executionPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"
    else:
        seedPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/forDiffTesting"
        executionPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/diffTesting"

    seedPaths = getSeeds(seedPath)
    counter = 0
    for folder, fileName in seedPaths:
        counter += 1
        if counter < args.startAt:
            continue
        nrOfsol = []
        status = []
        gc.collect()
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        for solver in solvers:
            try:
                m = Model().from_file(folder + "/" + fileName)
                sol = m.solveAll(solver=solver, time_limit=timeout, solution_limit=100)
                nrOfsol.append((sol, solver))
                status.append((m.status().exitstatus.name, solver))
            # except NotImplementedError:
            #     nrOfsol=0
            #
            #     while m.solve(solver=solver, time_limit=timeout) and nrOfsol<=100:
            #         m +=
            #     nrOfsol = (nrOfsol, solver)
            #     status.append((m.status().exitstatus, solver))
            except Exception as e:
                # crash
                print(colored("Crash" + str(e), "red", attrs=["bold"]))
                recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                            trace=traceback.format_exc(), errorName=str(e), solver=solver)

        if len(status) == 0 or len(nrOfsol) == 0:
            continue

        if len(set(nrOfsol)) <= 1 and len(set(status)) <= 1:
            continue

        recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status)

if __name__ == "__main__":
    __main__()
