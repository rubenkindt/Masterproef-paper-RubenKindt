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

from storm.utils.flatten_model import flatten_cpmpymodel
from storm.utils.randomness import Randomness
from termcolor import colored
from storm.fuzzer.helper_functions import export_mutants, enrich_true_and_false_nodes, \
    pick_true_and_false_nodes_at_random



def generate_mutants(cpmpy_Object, path_to_directory, maxDepth, maxAssert, seed, fuzzing_parameters):

    def generate_mutants_in_a_thread(cpmpy_Object, path_to_directory, seed, fuzzing_parameters):
        # We have to create a new randomness object here
        randomness = Randomness(seed)
        #model = cpmpy_Object.get_model()
        print("####### Generating mutants at location: " + colored(path_to_directory, "blue", attrs=["bold"]))
        get_all_truth_values_in_astVector(cpmpy_Object, maxDepth, maxAssert, fuzzing_parameters)
        print("####### Some stats: ")
        print("\t\tNumber of assertions = " + colored(str(cpmpy_Object.get_total_number_of_assertions()), "yellow", attrs=["bold"]))
        print("\t\tNumber of " + colored("TRUE ", "green", attrs=["bold"]) + "nodes = " + colored(str(len(cpmpy_Object.get_true_nodes())), "yellow", attrs=["bold"]))
        print("\t\tNumber of " + colored("FALSE ", "red", attrs=["bold"]) + "nodes = " + colored(str(len(cpmpy_Object.get_false_nodes())), "yellow", attrs=["bold"]))
        # Check if there is anything in the true and false nodes in smt_Object
        if len(cpmpy_Object.get_true_nodes()) > 0 or len(cpmpy_Object.get_false_nodes()) > 0:
            enrichment_steps = fuzzing_parameters["enrichment_steps"]
            number_of_mutants = fuzzing_parameters["number_of_mutants"]
            print("\t\tNumber of enrichment steps = " + colored(str(enrichment_steps), "yellow", attrs=["bold"]))
            print("\t\tNumber of mutants = " + colored(str(number_of_mutants), "yellow", attrs=["bold"]))
            print("\t\tMax Assert = " + colored(str(maxAssert), "yellow", attrs=["bold"]))
            print("\t\tMax Depth = " + colored(str(maxDepth), "yellow", attrs=["bold"]))
            print("####### Enriching the set of true and false nodes with more complex trees..")
            enrich_true_and_false_nodes(cpmpy_Object, enrichment_steps, randomness, maxDepth)
            print("\t\tNumber of " + colored("CONSTRUCTED_TRUE ", "green", attrs=["bold"]) + "nodes = " + colored(str(len(cpmpy_Object.get_true_constructed_nodes())), "yellow", attrs=["bold"]))
            print("\t\tNumber of " + colored("CONSTRUCTED_FALSE ", "red", attrs=["bold"]) + "nodes = " + colored(str(len(cpmpy_Object.get_false_constructed_nodes())), "yellow", attrs=["bold"]))

            print("####### Generating the mutants by picking true and false nodes..")
            mutants = pick_true_and_false_nodes_at_random(cpmpy_Object = cpmpy_Object,
                                                          number_of_mutants=number_of_mutants,
                                                          max_assertions=maxAssert,
                                                          randomness=randomness)
            print("####### Exporting mutants..")
            export_mutants(mutants, path_to_directory, cpmpy_Object)
            print("####### Done with exporting")
        else:
            print(colored("Nothing in TRUE or FALSE node. Nothing we can do here.", "red", attrs=["bold"]))

    generate_mutants_in_a_thread(cpmpy_Object,path_to_directory,seed,fuzzing_parameters)
    #process = multiprocessing.Process(target=generate_mutants_in_a_thread, args=(cpmpy_Object,path_to_directory,seed,fuzzing_parameters))
    #process.start()
    #process.join(fuzzing_parameters["mutant_generation_timeout"])
    #if process.is_alive():
        #process.terminate()
        #print(colored("TIMEOUT WHILE GENERATING MUTANTS", "red", attrs=["bold"]))
        #return 1
    return 0

def recursively_break_down_a_constraint_into_nodes(cpmpy_Object):
    cpmpy_Object, cpmpy_Object.flatModel = flatten_cpmpymodel(cpmpy_Object)
    return cpmpy_Object

'''def recursively_break_down_an_assertion_into_nodes(assertion, smt_Object, maxDepth):
    """
        Recusively evaluate a tree and append it in the respective
        true or false lists
    """
    # TODO:  Put a limit on how deep we want to go. Should be less than max python recursion limit
    #  Problem:
    #  Exception while computing tree depthargument 1: <class 'RecursionError'>: maximum recursion depth exceeded
    #  Fatal Python error: Cannot recover from stack overflow.
    if is_and(assertion) or is_or(assertion):
        tree_depth = get_tree_depth(assertion, maxDepth)
        if tree_depth < maxDepth:
            # Depth bound met
            smt_Object.append_to_all_nodes(assertion)
        children = assertion.children()
        for i, child in enumerate(children):
            if is_and(child) or is_or(child):
                recursively_break_down_an_assertion_into_nodes(child, smt_Object, maxDepth)
            else:
                # depth is already zero here
                smt_Object.append_to_all_nodes(child)
    else:
        # depth is already zero here
        smt_Object.append_to_all_nodes(assertion)
'''

def get_all_truth_values_in_astVector(cpmpy_Object, maxDepth, maxAssert, fuzzing_parameters):
    """
        Get truth values for all the leaves and sub-trees in the astVector
    """
    print("####### Breaking up assertions into nodes..")
    #ast = smt_Object.get_orig_ast()
    #for assertion in ast:
        # our model is already flattened
        #assertion_tree_depth = get_tree_depth(assertion, maxDepth, optimization=False)
        #if assertion_tree_depth > 99999:
        #    print(colored("\t\tTree depth is higher than the max recursion limit. Abort", "red", attrs=["bold"]))
        #    continue
        # Now we get truth values of nodes in the assertion
        #recursively_break_down_an_assertion_into_nodes(assertion, smt_Object, maxDepth)

    recursively_break_down_a_constraint_into_nodes(cpmpy_Object)

    print("####### Evaluating truth values for all nodes..")
    # Evaluate truth values of nodes in this assertion
    m = cpmpy.Model()
    m += cpmpy_Object.get_all_nodes()
    m.solve(solver="minizinc:chuffed", time_limit=fuzzing_parameters["solver_timeout"].total_seconds())

    for node in cpmpy_Object.get_all_nodes():
        try:
            if node.value():
                cpmpy_Object.append_true_node(node)
            else:
                cpmpy_Object.append_false_node(node)
        except Exception:
            continue

    '''for node in smt_Object.get_all_nodes():
        if model.eval(node, model_completion=True) == True:
            smt_Object.append_true_node(node)
        if model.eval(node, model_completion=True) == False:
            smt_Object.append_false_node(node)'''


"""
        print(colored("TRUE NODES", "green"))
        for node in smt_Object.get_true_nodes():
            print(model.eval(node, model_completion=True))
            if model.eval(node, model_completion=True) == False:
                print("PROBLEM !!!!!!")

        print(colored("\nFALSE NODES", "red"))
        for node in smt_Object.get_false_nodes():
            print(model.eval(node, model_completion=True))
            if model.eval(node, model_completion=True) == True:
                print("PROBLEM !!!!!!")
        """