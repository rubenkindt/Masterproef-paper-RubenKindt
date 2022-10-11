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

import multiprocessing

import cpmpy
from cpmpy.solvers import CPM_minizinc
from termcolor import colored
from cpmpy import *
from cpmpy.solvers.solver_interface import *

def check_satisfiability(cpmpy_Object, timeout):

    #def check_sat(cpmpy_Object, output):
    SolverLookup.get(cpmpy_Object.solver, cpmpy_Object.model).solve(timeout=timeout)
    solFound = cpmpy_Object.solver
    solverStatus = cpmpy_Object.model.status().exitstatus

    if solverStatus == ExitStatus.NOT_RUN:
        output = "unknown"
    elif solverStatus == ExitStatus.OPTIMAL:
        output = "sat"
    elif solverStatus == ExitStatus.FEASIBLE:
        output = "sat"
    elif solverStatus == ExitStatus.UNSATISFIABLE:
        output = "unsat"
    elif solverStatus == ExitStatus.ERROR:
        output = "error"
    elif solverStatus == ExitStatus.UNKNOWN:
        output = "unknown"
    else:
        output = "error"
    return output

    """output = multiprocessing.Array('c', b'unknown')
    process = multiprocessing.Process(target=check_sat, args=(cpmpy_Object, output))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        return "timeout"
    else:
        satisfiability = output.value.decode()
        return satisfiability"""


def convert_model_to_expression(model): # org name convert_ast_to_expression
    expression = model.constraints[0]
    for i in range(1, len(model.constraints)):
        expression = expression & model.constraints[i]
    return expression


def get_model(ast):
    s = Solver()
    s.add(ast)
    satis = s.check()
    if satis != sat:
        print(colored("Why are you sending me an unsat ast ?", "red"))
        raise Exception
    model = s.model()
    return model