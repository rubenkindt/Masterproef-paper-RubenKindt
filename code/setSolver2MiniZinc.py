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

    while "CMP_pysat" in lines:
        lines=lines.replace("CMP_pysat", "CPM_minizinc",-1)
    while "CMP_gurobi" in lines:
        lines = lines.replace("CMP_gurobi", "CPM_minizinc", -1)
    while "CPM_ortools" in lines:
        lines = lines.replace("CPM_ortools", "CPM_minizinc", -1)
    while ".solve()" in lines:
        location = lines.find(".solve()") + len(".solve(")
        lines = lines[:location] + "solver='minizinc:" + str(subsolver) + "'" + lines[location:]
    while ".solveAll()" in lines:
        location = lines.find(".solveAll()") + len(".solveAll(")
        lines = lines[:location] + "solver='minizinc:" + str(subsolver) + "'" + lines[location:]

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
        command = "timeout -s SIGKILL " + str(timeout) + "s " + str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName) + '" > "' + str(temp_file_path) + '"'
    else:
        command = str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName) + '" > "' + str(temp_file_path) + '"'

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
        filelist.append((filename, directory))

count=0
failed=0
#filelist= [("bowls_and_oranges.py", folder)]
for filename, folderpath in filelist:
    if not filename.endswith(".py"):
        continue
    if filename.__contains__("_WithMiniZinc_"):
        continue
    print("testing: "+str(folderpath)+str(filename))
    tempFolderPath, tempFileName = change_solver_to_MiniZinc(filename,folderpath,"chuffed")
    st=solver_runner("python3",tempFileName,tempFolderPath)
    if st.endswith("error"):
        failed+=1
    #check_if_already_MiniZinc(filename, filepath)
    count +=1

print(str(count)+" scripts found")
print(str(failed)+" scripts failed")

