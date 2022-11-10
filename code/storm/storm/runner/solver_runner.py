"""
Copyright 2020 MPI-SWS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import subprocess
import os

import cpmpy
from cpmpy import *
from cpmpy.solvers.solver_interface import ExitStatus
from termcolor import colored


def solver_runner(cp_file, temp_core_folder, timeout, solver):

    temp_file_name = "_output.txt"
    temp_file_path = temp_core_folder + temp_file_name

    model = cpmpy.Model().from_file(cp_file)

    #solFound = model.solve(solver=solver, time_limit=timeout.total_seconds())
    solFound = model.solve(solver=solver, time_limit=timeout.total_seconds())

    if model.status().exitstatus == ExitStatus.NOT_RUN:
        return "error" + " " + str("NOT_RUN")
    if model.status().exitstatus == ExitStatus.FEASIBLE:
        return "sat"
    if model.status().exitstatus == ExitStatus.OPTIMAL:
        return "sat"
    if model.status().exitstatus == ExitStatus.ERROR:
        return "error" + " " + str("solver ERROR")
    if model.status().exitstatus == ExitStatus.UNKNOWN:
        return "unknown"
    if model.status().exitstatus == ExitStatus.UNSATISFIABLE and not str(model.constraints).__contains__(" == 0 == 0"):
        return "unsat"
    else:
        return "unknown"


"""
def read_result(file_path, incremental):
    try:
        with open(file_path, 'r') as f:
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


    # Incremental mode
    if incremental == "yes":
        # If any result is unsat, return unsat
        for line in lines:
            if line.find("unsat") != -1:
                os.remove(file_path)
                return "unsat"

        for line in lines:
            if line.find("unknown") != -1:
                os.remove(file_path)
                return "unknown"

        os.remove(file_path)
        return "sat"



    if len(lines) > 0:
        if lines[0] == "sat" or lines[0] == "unsat" or lines[0] == "unknown":
            os.remove(file_path)
            return lines[0]
        else:
            return "error"
    else:
        os.remove(file_path)
        return "timeout"
"""