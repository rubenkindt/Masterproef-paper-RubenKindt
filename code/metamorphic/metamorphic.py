import os
import re
import shutil
import argparse
import copy
import traceback
import random

from cpmpy import *
from cpmpy.expressions.core import Comparison
from cpmpy.solvers.solver_interface import ExitStatus
from termcolor import colored


class metaModel():
    def __init__(self, solver=None, seedF=None):
        self.solver = solver
        self.seedFile = seedF
        self.metaRelations = list()
        if self.seedFile is None:
            self.origModel = None
            self.modifModel = None
        else:
            try:
                self.origModel = Model().from_file(self.seedFile)
                self.modifModel = copy.deepcopy(self.origModel)
            except Exception as e:
                pass

def create_file(data, path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--startAt', type=int, nargs='?', help='the bug-catcher likes to fill the swap space and crash')
    parser.add_argument('--seed', type=int, nargs='?', help='seed for random')
    arguments = parser.parse_args()

    if arguments.startAt is None:
        arguments.startAt = 0
    if arguments.seed is None:
        arguments.seed = 123
    random.seed(arguments.seed)
    timeout = 5 * 60  # 5 minutes
    if os.name == 'posix':
        seedPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/metamorphic"
        executionPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/metamorphic"
    else:
        seedPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples/metamorphic"
        executionPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/metamorphic"
    solvers = ["gurobi", "ortools"]  # "minizinc:gurobi", , "minizinc:chuffed"]

    seedPaths = getSeeds(seedPath)
    # random.shuffle(seedPaths)
    counter = 0
    for folder, fileName in seedPaths:
        counter += 1
        if counter < arguments.startAt:
            continue
        print("file " + str(counter) + "/" + str(len(seedPaths)) + ": " + fileName)
        mmodel = metaModel(solver=random.choice(solvers),seedF=folder + "/" + fileName)

        for i in range(1,10):
            mmodel.origModel.solve(solver=mmodel.solver, time_out=timeout)
            mutation(mmodel)
        
        """m = Model().from_file(folder + "/" + fileName)
        sol = m.solveAll(solver=solver, time_limit=timeout, solution_limit=100)
        print(colored("Crash" + str(e), "red", attrs=["bold"]))
        recordCrash(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                    trace=traceback.format_exc(), errorName=str(e), solver=solver)

        recordDiff(executionDir=executionPath, seedFolder=folder, seedName=fileName,
                   potentialNrDiff=nrOfsol, potentialStatusDiff=status)"""

def satMutation(satModel):
    metamorphicMutations = ["expandingAllDiff","AllDiff~(==)","expandAllEqual","AllEqual~(!=)","addNewVar2AllEqual",
                            "TrueAndCons", "FalseOrCons", "xorCons", "==1", "!=0", "sementicFusion+",
                            "addRandomIntRestrictions", "True->cons", "cons->True", "cons1<=>cons2"]
    choice = random.choice(metamorphicMutations)
    if choice == "expandingAllDiff":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if cons.name == 'alldifferent' and len(cons.args) <= 5:
                for i, arg1 in enumerate(cons.args):
                    for arg2 in cons.args[i:]:
                        newcons += [arg1 != arg2]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "AllDiff~(==)":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if cons.name == 'alldifferent' and len(cons.args) <= 5:
                for i, arg1 in enumerate(cons.args):
                    for arg2 in cons.args[i:]:
                        newcons += [~(arg1 == arg2)]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "expandAllEqual":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if cons.name == 'allequal':
                for arg1 in cons.args[1:]:
                    newcons += [cons.args[0] == arg1.name]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "AllEqual~(!=)":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if cons.name == 'allequal' and len(cons.args) <= 5:
                for i, arg1 in enumerate(cons.args):
                    for arg2 in cons.args[i:]:
                        newcons += [~(arg1 != arg2)]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "addNewVar2AllEqual":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if cons.name == 'allequal':
                if isinstance(cons.args[0], variables._BoolVarImpl):
                    var = boolvar()
                    allVar = cons.args+[var]
                    newcons += AllEqual(random.shuffle(allVar))

                if isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl):
                    var = intvar(lb=cons.args[0].lb, ub=cons.args[0].ub)
                    allVar = cons.args+[var]
                    newcons += AllEqual(random.shuffle(allVar))
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "TrueAndCons":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if random.random() > 0.5:
                newcons += [(cons) & True]
            else:
                newcons += [True & (cons)]
        satModel.modifModel = Model(newcons)
    elif choice == "FalseOrCons":
        newcons = []
        for cons in satModel.modifModel.constraints:
            if random.random() > 0.5:
                newcons += [False | (cons)]
            else:
                newcons += [(cons) | False]
        satModel.modifModel = Model(newcons)
    elif choice == "xorCons":
        newcons = []
        for cons in satModel.modifModel.constraints:
            newcons += Xor([False, True, True] + [cons])
        satModel.modifModel = Model(newcons)
    elif choice == "==1":
        newcons = []
        for cons in satModel.modifModel.constraints:
            newcons += [cons == 1]
        satModel.modifModel = Model(newcons)
    elif choice == "!=0":
        newcons = []
        for cons in satModel.modifModel.constraints:
            newcons += [cons != 0]
        satModel.modifModel = Model(newcons)
    elif choice == "sementicFusion+":
        newcons = []
        firstCons = None
        secCons = None
        for i, cons in enumerate(satModel.modifModel.constraints):
            if firstCons is None and isinstance(cons, Comparison) and \
                    (isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl)):
                firstCons = cons
                continue
            elif secCons is None and firstCons is not None and isinstance(cons, Comparison) and \
                    (isinstance(cons.args[0], variables._IntVarImpl) or isinstance(cons.args[0], variables._NumVarImpl)):
                secCons = cons
                continue
            newcons += cons

        if secCons is None:
            return

        x = firstCons.args[0]
        y = secCons.args[0]
        z = intvar(lb=x.lb + y.lb, ub=x.ub + y.ub)
        xr = (z - y)
        yr = (z - x)
        firstCons.args[0] = xr
        secCons.args[0] = yr
        newcons += firstCons & secCons  # will not do anything useful if both constraints come from the same file

        for i, cons in enumerate(newcons):
            if hasattr(cons.args[0], "name") and cons.args[0].name == x.name and random.random() < 0.1:
                cons.args[0] = xr
            if hasattr(cons.args[0], "name") and cons.args[1].name == x.name and random.random() < 0.1:
                cons.args[0] = xr
            if hasattr(cons.args[0], "name") and cons.args[0].name == y.name and random.random() < 0.1:
                cons.args[0] = yr
            if hasattr(cons.args[0], "name") and cons.args[1].name == y.name and random.random() < 0.1:
                cons.args[0] = yr

        satModel.modifModel = Model(newcons)
    elif choice == "addRandomIntRestrictions":
        newcons = []
        newcons += satModel.modifModel.constraints
        for loopVar in range(random.randint(1, 3)):
            i = intvar(lb=0, ub=10, shape=1, name="beep")
            if random.random() < 0.5:
                newcons += [i > random.randint(1, 9)]
            else:
                newcons += [i < random.randint(1, 9)]
        satModel.modifModel = Model(newcons)
    elif choice == "True->cons":
        newcons = []
        for cons in satModel.modifModel.constraints:
            temp = Model([True])
            if random.random() < 0.5:
                newcons += [(temp.constraints[0]).implies(cons)]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "cons->True":
        newcons = []
        for cons in satModel.modifModel.constraints:
            temp = Model([True])
            if random.random() < 0.5:
                newcons += [cons.implies(temp.constraints[0])]
            else:
                newcons += [cons]
        satModel.modifModel = Model(newcons)
    elif choice == "cons1<=>cons2":
        ToDo

if __name__ == "__main__":
    __main__()