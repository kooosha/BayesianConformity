#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:47:42 2019

@author: koosha
"""

import numpy as np
import os
from session import Session, Block

class DBN:

    @staticmethod           
    def fitAllPlayers(session, search_max = 60, decays = [1], win_values = [0]):
        num_trials = np.zeros(6)
        error = 1000 * np.ones([6, 1 + int (search_max / 6), len(decays), len(win_values)])
        best_error=np.zeros(6)
        best_decays = np.zeros(6)
        for player in range(6):
            for k in range(1, 1 + int (search_max / 6)):
                for d in range(len(decays)):
                    for w in range(len(win_values)):
                        error[player, k, d, w] = 0
                        for block in session.blocks:
                            if player >= block.choices.shape[1]:
                                continue
                            vals = np.round(session.values[block.item_ids, player], 2) + win_values[w]
                            alpha = decays[d] * k * np.sum(block.choices[0, :]) 
                            beta = decays[d] * k * (block.choices.shape[1] - np.sum(block.choices[0, :]))
                            for step in range(1, block.choices.shape[0]):
                                model_choice = int(alpha > beta)
                                error[player, k, d, w] += abs(model_choice - block.choices[step, player])
                                #update:
                                alpha += np.sum(block.choices[step, :]) 
                                beta += block.choices.shape[1] - np.sum(block.choices[step, :])
                                alpha = alpha * decays[d]
                                beta = beta * decays[d]
                                num_trials[player] +=1
            best_error[player] = int(search_max / 6) * len(win_values) * len(decays) * np.min(error[player, :, :]) / num_trials[player]
            best_k, best_d, best_w = np.unravel_index(error[player,:,:].argmin(), error[player,:, :].shape)         
            #print (player, best_k, decays[best_d], win_values[best_w], 1 - best_error[player]) 
            best_decays[player] = decays[best_d]
        return best_error, best_decays  
    
    @staticmethod           
    def loocvAllPlayers(session, search_max = 60, decays = [1], win_values = [0]):
        num_trials = np.zeros(6)
        error = 1000 * np.ones([len(decays)])
        best_error=np.zeros(6)
        for player in range(6):
            for left_block in session.blocks:
                if left_block.choices.shape[0] < 2:
                    continue
                if player >= left_block.getN():
                    continue
                num_trials[player] += left_block.choices.shape[0] - 1
                
                for d in range(len(decays)):
                    error[d] = 0
                    for block in session.blocks:
                        if player >= block.choices.shape[1]:
                            continue
                        if block.id == left_block.id:
                            continue
                        alpha = decays[d] *  np.sum(block.choices[0, :]) 
                        beta = decays[d] * (block.choices.shape[1] - np.sum(block.choices[0, :]))
                        for step in range(1, block.choices.shape[0]):
                            model_choice = int(alpha > beta)
                            error[d] += abs(model_choice - block.choices[step, player])
                            #update:
                            alpha += np.sum(block.choices[step, :]) 
                            beta += block.choices.shape[1] - np.sum(block.choices[step, :])
                            alpha = alpha * decays[d]
                            beta = beta * decays[d]
                best_d = np.argmin(error)         
                alpha = decays[best_d] * np.sum(left_block.choices[0, :]) 
                beta = decays[best_d] * (left_block.choices.shape[1] - np.sum(left_block.choices[0, :]))
                for step in range(1, left_block.choices.shape[0]):
                    model_choice = int(alpha > beta)
                    best_error[player] += abs(model_choice - left_block.choices[step, player])
                    #update:
                    alpha += np.sum(left_block.choices[step, :]) 
                    beta += left_block.choices.shape[1] - np.sum(left_block.choices[step, :])
                    alpha = alpha * decays[best_d]
                    beta = beta * decays[best_d]
            print (1 - best_error[player]/ num_trials[player]) 
        
        return best_error  

    @staticmethod           
    def allPlayersEnd(session):
        num_games = np.zeros(6)
        num_trials = np.zeros(6)
        for block in session.blocks:
            if block.choices.shape[0] < 2:
                continue
            num_games[ : block.choices.shape[1]] +=1
            num_trials[ : block.choices.shape[1]] += block.choices.shape[0]
        for p in range(6):
            print ((num_trials[p] - num_games[p])/num_trials[p])


session_list = [s_id for s_id in os.listdir('main/')]
avg = 0
decays = [1, .75, .25, .5, 0]
search_max = 12
win_values = [100, .5, .75, .9, 1, 1.1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 7.5, 8, 10, 15, 20, 0, .1, .2]

for i in range(20):
   session_id = session_list[i]
#    #print (session_id)
   s = Session(session_id)
   e = DBN.allPlayersEnd(s)
#    for p in range(6):
#        #print (p, 1 - e[p])
#        avg += e[p]
         
   