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

    for folder, fileName in allSoundnessBugs:
        if alreadyHasMin(folder):
            continue
        minimize(folder, fileName)

def alreadyHasMin(folder):
    """checks if folder has already a file starting with 'minimized'
    """

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith("minimized"):
                return True
    return False

def minimize(folderPath, fileName):
    m = cpmpy.Model().from_file(folderPath+"/"+fileName)
    unsatCons = musx(m.constraints)
    unsatModel = Model(unsatCons)
    unsatModel.to_file(folderPath+"/minimized")

    print()
    print(folderPath+"/"+fileName)
    for con in unsatModel.constraints:
        print(con)

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