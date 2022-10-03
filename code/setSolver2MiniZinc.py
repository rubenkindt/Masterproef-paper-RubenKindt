import subprocess
import os
from termcolor import colored

def check_if_already_MiniZinc(filename, folderpath):
    temp_file_name = filename.replace(".py", "")
    temp_file_name += "_output.txt"
    try:
        with open(folderpath+"/"+filename, 'r+') as f:
            lines = f.read(-1) #-1 = read all
            f.close()
    except:
        print(colored("CANT OPEN THE FILE:"+str(filename)+str(folderpath), "red", attrs=["bold"]))
        return "error"

    if 'SolverLookup.get("minizinc' in lines \
        or "SolverLookup.get('minizinc" in lines\
        or '.solve(solver="minizinc' in lines\
        or ".solve(solver='minizinc" in lines\
        or '.solve("minizinc' in lines\
        or ".solve('minizinc" in lines\
        or "CPM_minizinc(" in lines:
        print("found a miniZinc in "+str(filepath)+" "+str(filename))
        return True
    else:
        return False

def change_solver_to_MiniZinc(filename, folderpath, subsolver):
    try:
        with open(folderpath+"/"+filename, 'r') as f:
            lines = f.read(-1) #-1 = read all
            f.close()
    except:
        print(colored("CANT OPEN THE FILE:"+str(filename)+str(folderpath), "red", attrs=["bold"]))
        return None

    toFind = ["CMP_pysat(", "CMP_gurobi(", "CPM_ortools("]
    canChangeSolve=True
    for find in toFind:
        while find in lines:
            location = lines.find(find)
            end = lines[location:].find(")") + location
            modelVarName= lines[location+len(find):end]
            lines = lines[:location] + \
                    'CPM_minizinc(cpm_model=' + modelVarName + ", " + \
                    'subsolver="' + str(subsolver) + '"' + \
                    lines[end:]
            canChangeSolve=False

    while canChangeSolve and ".solve()" in lines:
        location = lines.find(".solve()") + len(".solve(")
        lines = lines[:location] + "solver='minizinc:" + str(subsolver) + "'" + lines[location:]

    while canChangeSolve and ".solveAll()" in lines:
        location = lines.find(".solveAll()") + len(".solveAll(")
        lines = lines[:location] + "solver='minizinc:" + str(subsolver) + "'" + lines[location:]

    # comment out solver specific attributes
    location = lines.find("ort_solver")
    while location != -1:
        if lines[location-15:location][::-1].find("#") == -1: #cant find "#" in front
            insertKardLocationRelative = lines[:location][::-1].find("\n") #search for first space and
            lines = lines[:location-insertKardLocationRelative] + "#" + lines[location-insertKardLocationRelative:] #add "#"
        location += len("ort_solver")
        premble = len(lines[:location])
        location = lines[location:].find("ort_solver")
        if location == -1:
            break
        location += premble

    #todo fix cases with solveAll(solver="stuff", display=None, time_limit=None, solution_limit=None) and solve(solver=None, time_limit=None)
    try:
        file = folderpath+"/"+"_WithMiniZinc_"+filename
        if os.path.isfile(file):
            os.remove(file)
        with open(file, 'w+') as f:
            f.write(lines)
            f.close()
            return folderpath, "_WithMiniZinc_"+filename
    except:
        print(colored("CANT OPEN THE FILE:"+str(filename)+str(folderpath), "red", attrs=["bold"]))
        return None

    return None

def read_result(file_path):
    try:
        with open(file_path, 'r+') as f:
            lines = f.read().splitlines()
            f.close()
            #print(lines)
    except:
        print(colored("CANT OPEN THE FILE: "+str(file_path), "red", attrs=["bold"]))
        return "error"

    for line in lines:
        if line.find("Parse Error") != -1:
            os.remove(file_path)
            return "parseerror"

        if line.find("Segmentation fault") != -1:
            os.remove(file_path)
            return "segfault"

        # java.lang.NullPointerException
        if line.find("NullPointerException") != -1:
            os.remove(file_path)
            return "nullpointer"

        if line.find("ASSERTION VIOLATION") != -1:
            os.remove(file_path)
            return "assertviolation"

        # java.lang.AssertionError
        if line.find("AssertionError") != -1:
            os.remove(file_path)
            return "assertviolation"

        if line.find("CAUGHT SIGNAL 15") != -1:
            os.remove(file_path)
            return "timeout"


    # Uninteresting problems
    for line in lines:
        if line.find("error") != -1 or line.find("unsupported reserved word") != -1:
            os.remove(file_path)
            return "error"
        if line.find("failure") != -1:
            os.remove(file_path)
            return "error"
    if len(lines) == 0:
        os.remove(file_path)
        return "timeout"

    if len(lines) > 0:
        if lines[0] == "sat" or lines[0] == "unsat" or lines[0] == "unknown":
            os.remove(file_path)
            return lines[0]
        else:
            os.remove(file_path)
            return "no (un)sat of unknown found"
    else:
        os.remove(file_path)
        return "timeout"
    return

def solver_runner(solver_path, fileName, folderPath, timeout=60, solver=None):

    temp_file_name = fileName.replace(".py", "")
    temp_file_name += "_output.txt"
    temp_file_path = folderPath + "/" + temp_file_name

    if os.name == 'posix':
        command = "cd " + folderPath + " ; " + "timeout -s SIGKILL " + str(timeout) + "s " + str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName) + '" > "' + str(temp_file_path) + '"'
    else:
        command = "cd " + folderPath + " & " + str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName) + '" > "' + str(temp_file_path) + '"'

    print(colored(command, "yellow"))

    p = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    terminal_output = p.stderr.read().decode()

    print(terminal_output)

    # Process terminal output first before result parsing
    if terminal_output.find("NULL pointer was dereferenced") != -1:
        return "nullpointer error"
    if terminal_output.find("assert") != -1 or terminal_output.find("AssertionError") != -1:
        return "assertviolation error"
    if terminal_output.find("segfault") != -1:
        return "segfault error"
    if terminal_output.find("Fatal failure") != -1:
        return "fatalfailure error"
    if terminal_output.find("ModuleNotFoundError") != -1:
        return "missing module error"
    if terminal_output.find("FileNotFoundError") != -1:
        return "FileNotFound Error"
    if terminal_output.find("Exception: MiniZinc solver returned with status 'Error'") != -1:
        return "MiniZinc Error"
    if terminal_output.find('AttributeError: "CPM_') != -1:
        return "solver does not have attribute error"
    if terminal_output.find("NotImplementedError") != -1:
        return "Not Implemented Error"
    if terminal_output.find("Error") != -1:
        return "error"

    solver_output = read_result(temp_file_path)
    try:
        os.remove(temp_file_path)
    except:
        pass
    return solver_output

if os.name == 'posix':
    folder = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples"
else:
    folder = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples"

filelist = []

for directory, dirs, filenames in os.walk(folder):
    for filename in filenames:
        if not filename.endswith(".py") or filename.__contains__("cpmpy_hakank") or filename.__contains__("_WithMiniZinc"):
            continue
        filelist.append((filename, directory))

count=0
failed=0
##filelist= [("bowls_and_oranges.py", folder)]
for nameOfFile, path in filelist:
    print(str(count+1) + "/" + str(len(filelist)) + "testing: "+str(path)+"/"+str(nameOfFile))
    #tempFolderPath, tempFileName = change_solver_to_MiniZinc(nameOfFile,path,"gecode")
    #st=solver_runner("python3",tempFileName,tempFolderPath)
    st=solver_runner("python3",nameOfFile,path)
    if st.__contains__("error"):
        failed+=1
    #check_if_already_MiniZinc(nameOfFile, path)
    count +=1

print(str(count)+" scripts found")
print(str(failed)+" scripts failed")