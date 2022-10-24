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
import os

import cpmpy
from cpmpy.transformations.flatten_model import *
from cpmpy.transformations.flatten_model import __is_flat_var
from termcolor import colored
from storm.utils.file_operations import create_smt2_file
import copy


def export_mutants(mutants, path, cpmpy_Object):

    dummy_ast = cpmpy_Object.get_dummy_ast()
    for i, ast in enumerate(mutants):
        new_ast = copy.deepcopy(dummy_ast)
        #new_ast.resize(0)

        new_ast += ast

        m = cpmpy.Model(new_ast)
        file_path = os.path.join(path, "mutant_" + str(i))
        m.to_file(file_path)

def supportedNegFunction(expr):
    # this is a patch to circumvent the bug of negating a global function
    # part come from negated_normal(expr): from cpmpy/transformations/flatten_model.py which may have changed when you read this

    return True

    if __is_flat_var(expr):
        return True

    elif isinstance(expr, Comparison):
        if expr.name == '==':
            return True
        elif expr.name == '!=':
            return True
        elif expr.name == '<=':
            return True
        elif expr.name == '<':
            return True
        elif expr.name == '>=':
            return True
        elif expr.name == '>':
            return True
        return True

    elif isinstance(expr, Operator):
        assert (expr.is_bool())

        if expr.name == 'and':
            return all([supportedNegFunction(arg) for arg in expr.args])
        elif expr.name == 'or':
            return all([supportedNegFunction(arg) for arg in expr.args])
        elif expr.name == '->':
            return supportedNegFunction(expr.args[0]) & supportedNegFunction(expr.args[1])
        else:
            raise NotImplementedError("negate_normal {}".format(expr))
            #return expr == 0  # can't do better than this...

    elif expr.name == 'xor':
        return supportedNegFunction(expr.args[0]) & supportedNegFunction(expr.args[1])

    else:
        # global...
        raise NotImplementedError("negate_normal {}".format(expr))
        #return False

def enrich_true_and_false_nodes(smt_object, enrichment_steps, randomness, max_depth):
    """
        Populate true_constructed_nodes and false_constructed_nodes with more complex trees.
        These trees are created by AND-ing or NOT-ing random nodes in true_nodes and false_nodes
        Obviously this is for the case when the original formula is SAT and we have a model
    """
    i = 0
    while i < enrichment_steps:
        # Choose an operator
        operator = randomness.random_choice(["and", "not"])

        # If NOT:
        #   choose a node from +ve or -ve pool
        #   If +ve node:
        #       Not(node) -> -ve pool
        #   If -ve node:
        #       Not(node) -> +ve pool
        if operator == "not":
            node_type = randomness.random_choice(["true", "false"])
            if len(smt_object.get_true_nodes()) == 0:
                node_type = "false"
            if len(smt_object.get_false_nodes()) == 0:
                node_type = "true"
            if node_type == "false":
                # if node type is false then negate it and add it to true constructed node
                false_node = randomness.random_choice(smt_object.get_false_nodes() + smt_object.get_false_constructed_nodes())
                if supportedNegFunction(false_node):
                    smt_object.append_true_constructed_node(~(false_node))
                else:
                    i-=1
            if node_type == "true":
                # if node type is true then negate it and add it to false constructed node
                true_node = randomness.random_choice(smt_object.get_true_nodes() + smt_object.get_true_constructed_nodes())
                if supportedNegFunction(true_node):
                    smt_object.append_false_constructed_node(~(true_node))
                else:
                    i-=1

        #   If AND:
        #       node1, node2 = choose from +ve or -ve pool
        #       If any node was from self.false:
        #           AND(node1, node2) -> self.false
        #       If no node was from self.false:
        #           AND(node1, node2) -> self.true
        if operator == "and":
            # Choose 2 nodes
            if len(smt_object.get_true_nodes()) > 0 and \
                    len(smt_object.get_false_nodes()) > 0:
                node_1_type = randomness.random_choice(["true", "false"])
                node_2_type = randomness.random_choice(["true", "false"])
            elif len(smt_object.get_true_nodes()) == 0:
                node_1_type = "false"
                node_2_type = "false"
            else:
                node_1_type = "true"
                node_2_type = "true"

            if node_1_type == "true":
                node_1 = randomness.random_choice(smt_object.get_true_nodes() + smt_object.get_true_constructed_nodes())
            else:
                node_1 = randomness.random_choice(smt_object.get_false_nodes() + smt_object.get_false_constructed_nodes())

            if node_2_type == "true":
                node_2 = randomness.random_choice(smt_object.get_true_nodes() + smt_object.get_true_constructed_nodes())
            else:
                node_2 = randomness.random_choice(smt_object.get_false_nodes() + smt_object.get_false_constructed_nodes())

            # Append if depth of both subtrees is <= depth
            #if get_tree_depth(node_1, max_depth) <= max_depth and get_tree_depth(node_2, max_depth) <= max_depth:
                # Depth bound of the newly contructed tree met
            if node_1_type == "false" or node_2_type == "false":
                smt_object.append_false_constructed_node((node_1) & (node_2))
            else:
                smt_object.append_true_constructed_node((node_1) & (node_2))
        i+=1


def pick_true_and_false_nodes_at_random(cpmpy_Object, number_of_mutants, max_assertions, randomness):
    """
        Depending on the size of the true_nodes and false_nodes
        Generate new ASTs
        Appends generation vector
    """
    mutants = list()    # A list of a list of assertions
    for i in range(number_of_mutants):
        mutant_file = list()    # A list of assertions
        number_of_assertions = randomness.get_a_non_prime_integer(max_assertions)
        j = 0
        while j < number_of_assertions:
            # Make a decision about picking either a simple node or an enriched node
            node_decision = randomness.random_choice(["simp", "simp","simp","enr","enr","enr","enr","enr","enr","enr"])

            if node_decision == "simp":
                # Pick a True or False node
                node_type = randomness.random_choice(["true", "false"])
                if len(cpmpy_Object.get_true_nodes()) == 0:
                    node_type = "false"
                if len(cpmpy_Object.get_false_nodes()) == 0:
                    node_type = "true"

                node = None
                if node_type == "true":
                    node = randomness.random_choice(cpmpy_Object.get_true_nodes())
                if node_type == "false":
                    node_2_neg = (randomness.random_choice(cpmpy_Object.get_false_nodes()))
                    if supportedNegFunction(node_2_neg):
                        node = ~node_2_neg
                    else:
                        j-=1
                        continue

                mutant_file.append(node)
            else:
                # pick a true of false constructed node
                constructed_node_type = randomness.random_choice(["true", "false"])
                if len(cpmpy_Object.get_true_constructed_nodes()) == 0:
                    constructed_node_type = "false"
                if len(cpmpy_Object.get_false_constructed_nodes()) == 0:
                    constructed_node_type = "true"
                if constructed_node_type == "true":
                    node = randomness.random_choice(cpmpy_Object.get_true_constructed_nodes())
                else:
                    node_2_neg = (randomness.random_choice(cpmpy_Object.get_false_constructed_nodes()))
                    if supportedNegFunction(node_2_neg):
                        node = ~node_2_neg
                    else:
                        j-=1
                        continue
                mutant_file.append(node)
            j+=1

        mutants.append(mutant_file)
    return mutants


# def get_tree_depth(assertion, maxDepth, optimization=True):
#     def get_tree_depth_iterative(tree):
#         # Iterative algorithm to find the height of a tree
#         if not is_and(tree) and not is_or(tree):
#             return 0
#         q = []
#         q.append(tree)
#         height = 0
#         while (True):
#             nodeCount = len(q)
#             if nodeCount == 0:
#                 return height
#             height += 1
#             # Optimization
#             # If the height has already exceeded the maxDepth, no need go any deeper
#             if optimization and height > maxDepth:
#                 return maxDepth + 1
#             while nodeCount > 0:
#                 node = q[0]
#                 q.pop(0)
#                 children = node.children()
#                 left_child = children[0]
#                 right_child = children[1]
#                 if is_and(left_child) or is_or(left_child):
#                     q.append(left_child)
#                 if is_and(right_child) or is_or(right_child):
#                     q.append(right_child)
#                 nodeCount -= 1
#     try:
#         depth = get_tree_depth_iterative(tree=assertion)
#     except Exception as e:
#         print(colored("Exception while computing tree depth" + str(e)))
#         return 1000000
#     return depth


'''def add_check_cp_using(exported_mutant_file_path, check_cp_using_option):
    """
        Replace the normal (check-sat) with (check-sat-using)
    """
    try:
        with open(exported_mutant_file_path, 'r') as f:
            lines = f.read().splitlines()
    except:
        print("#######" +  colored("ERROR OCCURRED IN 'check_sat_using'", "red", attrs=["bold"]))
    for i in range(len(lines)):
        if lines[i].find("(check-sat)") != -1:
            lines[i] = "(check-sat-using " + check_cp_using_option + ")" + "\n"
        else:
            lines[i] = lines[i] + "\n"
    file = open(exported_mutant_file_path, "w")
    file.writelines(lines)
    file.close()'''


"""
    Helper functions for generation of incremental instances
"""

def count_assertions(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    no_assertions = 0
    for line in lines:
        if line.find("(assert") != -1:
            no_assertions += 1

    return no_assertions

def get_factor(n, randomness):
    """
        returns a random factor for n and total number of factors
    """
    factors = list()
    for i in range(1, n):
        if n % i == 0:
            factors.append(i)
    return randomness.random_choice(factors), len(factors)


def insert_pushes_pops(mutant_file_paths, randomness):
    """
        Insert pushes and pops in the already exported files
    """
    print("####### Inserting pushes and pops")
    for file_path in mutant_file_paths:
        number_of_assertions = count_assertions(file_path)
        random_factor, number_of_factors = get_factor(number_of_assertions, randomness)
        interval = int(number_of_assertions / random_factor)
        lines = []
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        counter = interval
        header_text = True
        push_number = 0
        for i in range(len(lines)):
            if lines[i].find("(assert") != -1:
                header_text = False
                if counter == interval:
                    lines[i] = "\n(push " + str(interval) + ")" + "\n  (assert\n"
                    counter = 1
                    push_number += 1
                else:
                    counter += 1
                    lines[i] = "  " + lines[i] + "\n"
            else:
                if header_text or lines[i].find("check-sat") != -1:
                    lines[i] = lines[i] + "\n"
                else:
                    lines[i] = "  " + lines[i] + "\n"
            if lines[i].find("(push") != -1 and push_number > 1:
                lines[i] = "(pop " + str(randomness.get_random_integer(0, interval)) + ")\n" + lines[i]
                lines[i] = "(check-sat)\n" + lines[i]
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()

    print("####### Insertion of pushes and pops successful")
