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
from cpmpy.solvers.solver_interface import SolverInterface, SolverStatus, ExitStatus

def check_satisfiability(cpmpy_Object, timeout):

    def check_sat(cpmpy_Object, output):
        solFound = cpmpy_Object.solver.solve(time_limit=timeout)
        status = cpmpy_Object.solver.status()

        if status == ExitStatus.NOT_RUN:
            output.value = "unknown".encode()
        elif status == ExitStatus.OPTIMAL:
            output.value = "sat".encode()
        elif status == ExitStatus.FEASIBLE:
            output.value = "sat".encode()
        elif status == ExitStatus.UNSATISFIABLE:
            output.value = "unsat".encode()
        elif status == ExitStatus.ERROR:
            raise Exception("MiniZinc solver returned with status 'Error'")
        elif status == ExitStatus.UNKNOWN:
            output.value = "unknown".encode()
        else:
            raise NotImplementedError  # a new status type was introduced, please report on github


    output = multiprocessing.Array('c', b'unknown')
    process = multiprocessing.Process(target=check_sat, args=(cpmpy_Object, output))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        return "timeout"
    else:
        satisfiability = output.value.decode()
        return satisfiability


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