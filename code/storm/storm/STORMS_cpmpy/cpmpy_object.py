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
import cpmpy
from termcolor import colored
from storm.runner.cpmpy_python_api import check_satisfiability, convert_model_to_expression, get_model
import copy

class cpmpyObject(object):
    def __init__(self, file_path, path_to_mutant_folder):
        self.path_to_orig_smt_file = file_path
        self.path_to_mutant_folder = path_to_mutant_folder
        self.orig_satisfiability = None
        self.readable = True
        self.orig_ast = None
        self.true_nodes = list()
        self.false_nodes = list()
        self.all_nodes = list()
        self.dummy_ast = None
        self.true_constructed_nodes = list()
        self.false_constructed_nodes = list()
        self.total_number_of_assertions = 0
        self.model = None
        self.flatModel = None
        self.solution = None
        self.solver = "minizinc:chuffed"


        try:
            self.orig_ast = cpmpy.Model.from_file(file_path).constraints
            self.dummy_ast = cpmpy.Model.from_file(file_path).constraints
            self.model = cpmpy.Model.from_file(file_path)
            self.total_number_of_assertions = len(self.model.constraints)

        except:
            print(colored("Exception while parsing cpmpy pickled file", "red", "on_white"))
            self.readable = False
        self.negated_model = None



    def get_readeble(self): #old name get_validity
        return self.readable
    def get_orig_ast(self):
        return self.orig_ast
    def get_negated_ast(self):
        return self.negated_model
    def get_dummy_ast(self):
        return self.dummy_ast
    def get_orig_satisfiability(self):
        return self.orig_satisfiability
    def append_true_node(self, node):
        self.true_nodes.append(node)
    def append_false_node(self, node):
        self.false_nodes.append(node)
    def append_true_constructed_node(self, node):
        self.true_constructed_nodes.append(node)
    def append_false_constructed_node(self, node):
        self.false_constructed_nodes.append(node)
    def append_to_all_nodes(self, node):
        self.all_nodes.append(node)
    def get_all_nodes(self):
        return self.all_nodes
    def get_true_nodes(self):
        return self.true_nodes
    def get_false_nodes(self):
        return self.false_nodes
    def get_true_constructed_nodes(self):
        return self.true_constructed_nodes
    def get_false_constructed_nodes(self):
        return self.false_constructed_nodes
    def get_total_number_of_assertions(self):
        return self.total_number_of_assertions
    def get_flatModel(self):
        return self.flatModel


    def check_satisfiability(self, timeout):
        self.orig_satisfiability = check_satisfiability(self, timeout)
        # SAT
        if self.orig_satisfiability == "sat":
            print(colored(self.path_to_orig_smt_file, "blue", attrs=["bold"]) + ": " + colored(self.orig_satisfiability, "green", attrs=["bold"]))
        # UNSAT
        if self.orig_satisfiability == "unsat":
            print(colored(self.path_to_orig_smt_file, "blue", attrs=["bold"]) + ": " + colored(self.orig_satisfiability, "red", attrs=["bold"]), end="")
            self.negated_model = ~(convert_model_to_expression(self.model)) #this stuff may break
            if check_satisfiability(self.negated_model, timeout) == "timeout":
                print(colored("   timeout on unsat -> sat file", "red"))
                self.readable = False
            else:
                print(colored("   successfully converted unsat -> sat", "green"))
        # TIMEOUT
        if self.orig_satisfiability == "timeout":
            print(colored(self.path_to_orig_smt_file, "blue", attrs=["bold"]) + ": " + colored(self.orig_satisfiability, "red", "on_white"))


    def get_model(self):
        if self.orig_satisfiability == "sat":
            self.model = get_model(self.orig_ast)
        if self.orig_satisfiability == "unsat":
            self.model = get_model(self.negated_model)
        return self.model




