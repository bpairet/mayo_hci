'''
Contains a bunch of functions that are written to help using MAYO with 
adaptive contstraints
'''

'''
 MAYO pipeline, from Pairet et al. 2020
    Copyright (C) 2020, Benoit Pairet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
'''

'''
import pick

import numpy as np


def create_list_lambda_d_and_p(list_lambda_d,list_lambda_p,make_all_possible_pairs=False): 
    list_lambda_d_and_p = []
    if make_all_possible_pairs:
        reversing__lambda_p = False 
        for _lambda_p in list_lambda_p:
            for ii in range(len(list_lambda_d)):
                if reversing__lambda_p:
                    _lambda_d = list_lambda_d[len(list_lambda_d)-1-ii]
                else:
                    _lambda_d = list_lambda_d[ii]
                list_lambda_d_and_p.append((float(_lambda_d),float(_lambda_p)))
            if reversing__lambda_p:
                reversing__lambda_p = False
            else:
                reversing__lambda_p = True
    else:
        assert(len(list_lambda_d) == len(list_lambda_p)),"list_lambda_d and list_lambda_p must be of same length when make_all_possible_pairs==False"                       
        for i,_ in enumerate(list_lambda_p):
            list_lambda_d_and_p.append((float(list_lambda_d[i]),float(list_lambda_p[i])))
    return list_lambda_d_and_p

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def ask_what_to_do_adaptive_lasso(ratio_noise, ratio_d_and_p):
    if isnotebook():
        is_return = True
    else:
        title = 'What do you wanna do ? '
        options = ['return', 'change ratio_noise', 'change ratio_d_and_p','continue']
        answer, _ = pick.pick(options, title)
        if answer == 'continue':
            is_return = False
        elif answer == 'return':
            is_return = True
        elif answer == 'change ratio_noise':
            is_return = False
            print('please select the new value of ratio_noise, current on is '+ str(ratio_noise))
            try:
                ratio_noise = float(input("New value of ratio_noise:"))
            except ValueError:
                ratio_noise = float(input("ratio_noise must be a float:"))
            print('ratio_noise is set to ' + str(ratio_noise))
        elif answer == 'change ratio_d_and_p':
            is_return = False
            print('please select the new value of ratio_d_and_p, current on is '+ str(ratio_d_and_p))
            try:
                ratio_d_and_p = float(input("New value of ratio_d_and_p:"))
            except ValueError:
                ratio_d_and_p = float(input("ratio_d_and_p must be a float:"))
            print('ratio_d_and_p is set to ' + str(ratio_d_and_p))
    return is_return, ratio_noise, ratio_d_and_p


def ask_what_to_do_adaptive_constraint(tau_disk,tau_planet):
    if isnotebook():
        is_return = True
    else:
        title = 'What do you wanna do ? '
        options = ['return', 'change tau_disk', 'change tau_planet','continue']
        answer, _ = pick.pick(options, title)
        if answer == 'continue':
            is_return = False
        elif answer == 'return':
            is_return = True
        elif answer == 'change change tau_disk':
            is_return = False
            print('please select the new value of change tau_disk, current on is '+ str(tau_disk))
            try:
                tau_disk = float(input("New value of change tau_disk:"))
            except ValueError:
                tau_disk = float(input("change tau_disk must be a float:"))
            print('tau_disk is set to ' + str(tau_disk))
        elif answer == 'change tau_planet':
            is_return = False
            print('please select the new value of tau_planet, current on is '+ str(tau_planet))
            try:
                tau_planet = float(input("New value of tau_planet:"))
            except ValueError:
                tau_planet = float(input("tau_planet must be a float:"))
            print('tau_planet is set to ' + str(tau_planet))
    return is_return, tau_disk, tau_planet

def ask_what_to_do_adaptive(regularization_disk,regularization_planet):
    if isnotebook():
        is_return = True
    else:
        title = 'What do you wanna do ? '
        options = ['return', 'change regularization_disk', 'change regularization_planet','continue']
        answer, _ = pick.pick(options, title)
        if answer == 'continue':
            is_return = False
        elif answer == 'return':
            is_return = True
        elif answer == 'change regularization_disk':
            is_return = False
            print('please select the new value of change regularization_disk, current on is '+ str(regularization_disk))
            try:
                regularization_disk = float(input("New value of change regularization_disk:"))
            except ValueError:
                regularization_disk = float(input("regularization_disk tau_disk must be a float:"))
            print('regularization_disk is set to ' + str(regularization_disk))
        elif answer == 'change regularization_planet':
            is_return = False
            print('please select the new value of regularization_planet, current on is '+ str(regularization_planet))
            try:
                regularization_planet = float(input("New value of regularization_planet:"))
            except ValueError:
                regularization_planet = float(input("regularization_planet must be a float:"))
            print('regularization_planet is set to ' + str(regularization_planet))
    return is_return, regularization_disk, regularization_planet


def ask_what_to_do_only_disk(list_constraint_disk,current_saving_dir):
    title = 'Algo done, check '+current_saving_dir+' and tell me what to do \n current list_constraint_disk is '+str(list_constraint_disk)
    options = ['return', 'change regularization_disk']
    answer, _ = pick.pick(options, title)
    if answer == 'return':
        is_return = True
    elif answer == 'change regularization_disk':
        is_return = False
        lower = float(input('please select the new lower limit of disk regularization:  \n current list_constraint_disk is '+str(list_constraint_disk))  ) 
        upper = float(input('please select the new upper limit of disk regularization:  \n current list_constraint_disk is '+str(list_constraint_disk)) )  
        number = float(input('please select the new number of disk regularization:  \n current list_constraint_disk is '+str(list_constraint_disk))    )
        list_constraint_disk = np.linspace(lower,upper,number)
    return is_return, list_constraint_disk


def ask_what_to_do_only_planet(list_constraint_planet,current_saving_dir):
    title = 'Algo done, check '+current_saving_dir+' and tell me what to do \n current list_constraint_planet is '+str(list_constraint_planet)
    options = ['return', 'change regularization_disk']
    answer, _ = pick.pick(options, title)
    if answer == 'return':
        is_return = True
    elif answer == 'change regularization_disk':
        is_return = False
        lower = float(input('please select the new lower limit of disk regularization:  \n current list_constraint_planet is '+str(list_constraint_planet))  ) 
        upper = float(input('please select the new upper limit of disk regularization:  \n current list_constraint_planet is '+str(list_constraint_planet)) )  
        number = float(input('please select the new number of disk regularization:  \n current list_constraint_planet is '+str(list_constraint_planet))    )
        list_constraint_planet = np.linspace(lower,upper,number)
    return is_return, list_constraint_planet
'''
