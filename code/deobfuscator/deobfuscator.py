import os
import cpmpy
from termcolor import colored
from musx import *

def __main__():
    if os.name == 'posix':
        homePath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/soundness"
    else:
        homePath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/soundness"

    allSoundnessBugs = []
    for root, dirs, files in os.walk(homePath):
        for file in files:
            if file.startswith('mutant'):
                allSoundnessBugs.append((root,file))

    allSoundnessBugs.sort()

    for folder, fileName in allSoundnessBugs:
        if alreadyHasMin(folder):
            continue
        minimize(folder, fileName)

def minimize(folderPath, fileName):
    m = cpmpy.Model().from_file(folderPath+"/"+fileName)
    try:
        unsatCons = musx(m.constraints)
    except Exception as e:
        return
    unsatModel = Model(unsatCons)
    unsatModel.to_file(folderPath+"/minimized")

    print()
    print(folderPath+"/"+fileName)

    constraints = ""
    for con in unsatModel.constraints:
        constraints += str(con)
    print(constraints)

    if not constraints.__contains__(" == 0 == 0"):
        storeIntrestingFile(folderPath, fileName, constraints)

def storeIntrestingFile(folderPath, fileName, constraints):
    if os.name == 'posix':
        intPath = "/home/user/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/interesting.txt"
    else:
        intPath = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/results/storm/temp/interesting.txt"

    strin = folderPath+"/"+fileName+"\n"+constraints+"\n"
    file = open(intPath,"a")
    file.write(strin)
    file.close()

def alreadyHasMin(folder):
    """checks if folder has already a file starting with 'minimized'
    """

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith("minimized"):
                return True
    return False

def checkAgain(folder, fileName):
    file = folder+"/"+fileName

    m = Model().from_file(file)
    m.solve(solver="ortools")
    m.status()
    #print(m.status())

    m2 = Model().from_file(file)
    m2.solve(solver="minizinc:chuffed")
    #print(m2.status())
    
    assert m.status().exitstatus.name != m2.status().exitstatus.name

__main__()
