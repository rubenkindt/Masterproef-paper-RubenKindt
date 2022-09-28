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
        print(colored("CANT OPEN THE FILE:" +str(filename)+ str(folderpath), "red", attrs=["bold"]))
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

def read_result(file_path):
    try:
        with open(file_path, 'r+') as f:
            lines = f.read().splitlines()
            #print(lines)
    except:
        print(colored("CANT OPEN THE FILE", "red", attrs=["bold"]))
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
            return "error"
    else:
        os.remove(file_path)
        return "timeout"

def solver_runner(solver_path, smt_file, temp_core_folder, timeout=60, solver=None):

    temp_file_name = smt_file.replace(temp_core_folder, "")
    temp_file_name = temp_file_name.replace(".py", "")
    temp_file_name += "_output.txt"
    temp_file_path = temp_core_folder + "/" + temp_file_name

    if os.name == 'posix':
        command = "timeout " + str(timeout) + "s " + str(solver_path) + ' "' + str(temp_core_folder) + "/" + str(smt_file) + '" > "' + str(temp_file_path) + '"'
    else:
        command = str(solver_path) + ' "' + str(temp_core_folder) + "/" + str(smt_file) + '" > "' + str(temp_file_path) + '"'

    print(colored(command, "yellow"))

    p = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    terminal_output = p.stderr.read().decode()

    print(terminal_output)

    # Process terminal output first before result parsing
    if terminal_output.find("NULL pointer was dereferenced") != -1:
        return "nullpointer"
    if terminal_output.find("assert") != -1 or terminal_output.find("AssertionError") != -1:
        return "assertviolation"
    if terminal_output.find("segfault") != -1:
        return "segfault"
    if terminal_output.find("Fatal failure") != -1:
        return "fatalfailure"
    if terminal_output.find("ModuleNotFoundError") != -1:
        return "missing module"
    if terminal_output.find("FileNotFoundError") != -1:
        return "FileNotFoundError"
    if terminal_output.find("Error") != -1:
        return "error"

    solver_output = read_result(temp_file_path)
    try:
        os.remove(temp_file_path)
    except:
        pass
    return solver_output

if os.name == 'posix':
    folder = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples/"
else:
    folder = "C:\\Users\\ruben\\Desktop\\Thesis\\Masterproef-paper\\code\\examples"

filelist = []
count=0
for directory, dirs, filenames in os.walk(folder):
    for filename in filenames:
        filelist.append((filename, directory))


for filename, filepath in filelist:
    if filename.endswith(".py"):
        solver_runner("python3",filename,filepath)
        #check_if_already_MiniZinc(filename, filepath)
        count +=1

print(str(count)+" scripts found")
#ret = solver_runner("python3", "C:\\Users\\ruben\\Desktop\\Thesis\\Masterproef-paper\\code\\examples\\nqueens.py",
#              "C:\\Users\\ruben\\Desktop\\Thesis\\Masterproef-paper\\code\\examples\\",
#              10, incremental="no", solver="CPM")
#print(ret)
