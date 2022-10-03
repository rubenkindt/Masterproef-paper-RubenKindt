import os
import termcolor

def solver_runner(solver_path, fileName, folderPath, timeout=60, solver=None):

    if os.name == 'posix':
        command = "cd " + folderPath + " ; " + "timeout -s SIGKILL " + str(timeout) + "s " + str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName)
    else:
        command = "cd " + folderPath + " & " + str(solver_path) + ' "' + str(folderPath) + "/" + str(fileName)

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
        return "FileNotFound error"
    if terminal_output.find("Exception: MiniZinc solver returned with status 'Error'") != -1:
        return "MiniZinc Error"
    if terminal_output.find('AttributeError: "CPM_') != -1:
        return "solver does not have attribute error"
    if terminal_output.find("NotImplementedError") != -1:
        return "Not Implemented error"
    if terminal_output.find("Error") != -1:
        return "error"

    return "pass"

if os.name == 'posix':
    folder = "/home/user/Desktop/Thesis/Masterproef-paper/code/examples"
else:
    folder = "C:/Users/ruben/Desktop/Thesis/Masterproef-paper/code/examples"

filelist = []
for directory, dirs, filenames in os.walk(folder):
    for filename in filenames:
        if not filename.endswith(".py") or filename.__contains__("cpmpy_hakank") or \
            filename.__contains__("_WithMiniZinc") or filename.__contains__("Pickled"):
            continue
        filelist.append((filename, directory))

count = 0
##filelist= [("bowls_and_oranges.py", folder)]
for nameOfFile, path in filelist:
    print(str(count+1) + "/" + str(len(filelist)) + "testing: "+str(path)+"/"+str(nameOfFile))
    solver_runner("Python3",nameOfFile,path)
    count +=1
