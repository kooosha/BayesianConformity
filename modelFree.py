#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 02:25:23 2019

@author: koosha
"""
from __future__ import division, print_function

import numpy as np


import os
from session import Session, Block
import pickle

class ModelFree:  
    @staticmethod           
    def fitAllPlayers(session, win_values):
        all_errors = {}
        all_params = {}
        for player in range(6):
            best_error = 100000
            best_win_val = 0 
            best_bias = 0
            best_beta = 0
            for win_val in win_values:
                for bias in range(1, 100, 2):
                    for beta in range(0, 20, 2):
                        error = 0    
                        for block in session.blocks:
                            if player >= block.getN():
                                continue
                            Q_vals = np.round(session.values[block.item_ids, player] + win_val, 2)
                            for step in range(1, block.choices.shape[0]):
                                alpha = 1 / (bias + beta * (step-1))
                                Q_vals[int(block.choices[step-1, player])] = (1-alpha) * Q_vals[int(block.choices[step-1, player])]  
                                action = np.argmax(Q_vals)
                                error += abs(action - block.choices[step, player])
                        if error < best_error:
                            best_error = error
                            best_win_val = win_val
                            best_bias = bias
                            best_beta = beta
            num_trials = 0
            for block in session.blocks:
                num_trials+= block.choices.shape[0] - 1
            print (player, best_win_val, best_bias, best_beta, 1 - best_error/ num_trials)
            all_errors[player] = best_error/num_trials
            all_params[player] = (best_win_val, best_bias, best_beta)
        return all_errors, all_params

    @staticmethod           
    def fitAllPlayersLoocv(session, win_values):
        all_errors = {}
        for player in range(6):
            all_errors_player = 0
            num_trials = 0
            for left_block in session.blocks:
                if left_block.choices.shape[0] < 2:
                    continue
                if player >= left_block.getN():
                    continue
                num_trials+= left_block.choices.shape[0] - 1
                best_error = 100000
                best_win_val = 0 
                best_bias = 0
                best_beta = 0 
                for win_val in win_values:
                    for bias in range(1, 100, 20):
                        for beta in range(0, 20, 5):
                            error = 0    
                            for block in session.blocks:
                                if block.id == left_block.id:
                                    continue
                                if player >= block.getN():
                                    continue
                                Q_vals = np.round(session.values[block.item_ids, player] + win_val, 2)
                                for step in range(1, block.choices.shape[0]):
                                    alpha = 1 / (bias + beta * (step-1))
                                    Q_vals[int(block.choices[step-1, player])] = (1-alpha) * Q_vals[int(block.choices[step-1, player])]  
                                    action = np.argmax(Q_vals)
                                    error += abs(action - block.choices[step, player])
                            if error < best_error:
                                best_error = error
                                best_win_val = win_val
                                best_bias = bias
                                best_beta = beta
                                
                # Calculate error on the left one:
                Q_vals = np.round(session.values[left_block.item_ids, player] + best_win_val, 2)
                error = 0
                for step in range(1, left_block.choices.shape[0]):
                    alpha = 1 / (best_bias + best_beta * (step-1))
                    Q_vals[int(left_block.choices[step-1, player])] = (1-alpha) * Q_vals[int(left_block.choices[step-1, player])]  
                    action = np.argmax(Q_vals)
                    error += abs(action - left_block.choices[step, player]) 
                all_errors_player += error    
                               
            print (player, 1 - all_errors_player/ num_trials)
            all_errors[player] = all_errors_player/ num_trials
        return all_errors    

win_values = [0, .1, .2, .5, .75, .9, 1, 1.1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 7.5]
win_values = [7.5, 6, 3, 1.5, 1.25, 1.1, 1, .9, .5, .2, .1, 0]
session_list = [s_id for s_id in os.listdir('main/')]
all_es = {}
all_ps = {}
for i in range(len(session_list)):
    session_id = session_list[i]
    #print (session_id)
    s = Session(session_id)
    e, p = ModelFree.fitAllPlayers(s, win_values)     
    for player in range(6):
        all_es[(session_id, player)] = e[player]
        all_ps[(session_id, player)] = p[player]

#pickle.dump (all_ps, open('free_params', 'wb'))
#pickle.dump (all_es, open('free_errors', 'wb'))

for i in range(len(session_list)):
    session_id = session_list[i]
    s = Session(session_id)
    e = ModelFree.fitAllPlayersLoocv(s, win_values)     
    for player in range(6):
        all_es[(session_id, player)] = e[player]
        
#pickle.dump (all_es, open('free_errors_loocv', 'wb'))

