from cpmpy import *
import os, re, traceback, shutil
from termcolor import colored

def solveWith(lstConstraints, solver, timeout=None):
    """ parameters
        list of constraints,
        which solver you want to be solving
        timeout in seconds
        result
        the resulting model
    """
    m = cpmpy.Model(lstConstraints)
    m.solve(solver=solver, time_limit=timeout)

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


def recordDiff(executionDir, seedFolder, seedName, solvers, difference):
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
    safeErrorType = re.sub('[^a-zA-Z0-9 ]', '', " ".join(solvers))  # remove all non (a-z A-Z 0-9 and " ") characters
    path_to_bug_dir = os.path.join(path_to_crash_folder, safeErrorType + str(number_of_directories))
    os.mkdir(path_to_bug_dir)

    shutil.copy2(seedFolder + "/" + seedName, path_to_bug_dir)

    crash_logs = "seed: " + str(seedFolder + "/" + seedName) + "\n"
    crash_logs += "disagreeing solvers: " + str(solvers) + "\n"
    if difference == "nr+status":
        crash_logs += "difference is both nr of solution and status" + "\n"
    elif difference == "nr":
        crash_logs += "difference is amount of solution" + "\n"
    elif difference == "status":
        crash_logs += "difference is status" + "\n"

    create_file(crash_logs, os.path.join(path_to_bug_dir, "crash_logs.txt"))


def __main__():
    solvers = ["ortools", "gurobi"] #"minizinc:gurobi", , "minizinc:chuffed"]
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
        nrOfsol = []
        status = []
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        for solver in solvers:
            m = Model().from_file(folder + "/" + fileName)
            try:
                sol = m.solveAll(solver=solver, time_limit=timeout, solution_limit=100)
                nrOfsol.append((sol, solver))
                status.append((m.status().exitstatus, solver))
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

        solverWithDiffNr = []
        solverWithDiffstatus = []
        for item in set(nrOfsol):
            solverWithDiffNr.append(item[1])
        for item in set(nrOfsol):
            solverWithDiffstatus.append(item[1])

        if len(solverWithDiffNr) > 1 and len(solverWithDiffstatus) > 1:
            difference = "nr+status"
            diffSolvers = solverWithDiffNr + solverWithDiffstatus
        elif len(solverWithDiffNr) > 1:
            difference = "nr"
            diffSolvers = solverWithDiffNr
        elif len(solverWithDiffstatus) > 1:
            difference = "status"
            diffSolvers = solverWithDiffstatus
        else:
            raise Exception("this should not happen")

        recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName, solvers=diffSolvers,
                   difference=difference)

if __name__ == "__main__":
    __main__()
