#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 02:25:23 2019

@author: koosha
"""
from __future__ import division, print_function

import numpy as np
import scipy.io as io
from pomdp import POMDP
from ipomdp import IPOMDP
from scipy import special

import os
from session import Session, Block
import pickle

from dbn import DBN 

class POMDPFit:
    
    @staticmethod           
    def fitAllPlayers(session, decays, win_values, depth = 1, T_dail = .25, discount = .5, num_round = 25):
        all_end_errors = {}
        all_params = {}
        errors = np.zeros ([6, len(win_values), len(decays)])
        num_trials = np.zeros(6)
        for block in session.blocks:
            if block.choices.shape[0] < 2:
                continue
            num_players = block.getN()
            for p in range(num_players):
                vals = np.round(session.values[block.item_ids, p], 2)
                for w in range(len(win_values)):
                    for dec in range(len(decays)):
                        if depth == 1:
                            qvals = POMDP.loadPomdpQVals(num_players, vals, T_fail, decays[dec], discount, num_round, 6, win_values[w])
                        else:
                            qvals = IPOMDP.loadPomdpQVals(num_players, vals, T_fail, decays[dec], discount, num_round, 6, depth, win_values[w])
                        err = POMDPFit.findPlayerBlockErrors(block, p, qvals, decays[dec], win_values[w], discount, depth)
                        errors[p, w, dec] += err
                num_trials[p] += block.choices.shape[0] - 1
        for p in range(6):
            avg_err = errors[p, :, :] / num_trials[p]
            #print (p, win_values[np.unravel_index(avg_err.argmin(), avg_err.shape)[0]], decays[np.unravel_index(avg_err.argmin(), avg_err.shape)[1]], 1 - np.min(avg_err), end = " ")
            all_errors [p] = np.min(avg_err)
            vals = np.round(session.values[block.item_ids, p], 2)
            w, dec = np.unravel_index(avg_err.argmin(), avg_err.shape)
            if depth == 1:
                qvals = POMDP.loadPomdpQVals(num_players, vals, T_fail, decays[dec], discount, num_round, 6, win_values[w])
            else:
                qvals = IPOMDP.loadPomdpQVals(num_players, vals, T_fail, decays[dec], discount, num_round, 6, depth, win_values[w])
            end_error = 0          
            for block in session.blocks:
                if p < block.getN():
                    end_error += POMDPFit.findPlayerEndPredict(block, p, qvals,  decays[dec])
            #print (1 - end_error/num_trials[p])
            all_params [p] = (win_values[w], decays[dec])
            errors [p, :, :] = errors[p,:, :] / num_trials[p]
            all_end_errors[p] = end_error / num_trials[p]
        return errors, all_params, all_end_errors
    
    @staticmethod           
    def fitAllPlayersLoocv(session, decays, win_values, depth = 1, T_dail = .25, discount = .5, num_round = 25):
        loocv_error = np.zeros(6)
        all_errors, params, all_end_errors = POMDPFit.fitAllPlayers(session, decays, win_values, depth, T_dail, discount, num_round) 
        num_trials = np.zeros(6)
        for player in range(6):
            for block in session.blocks:
                if block.choices.shape[0] < 2:
                    continue
                if player >= block.getN():
                    continue       
                num_trials[player] += block.choices.shape[0] - 1
        for player in range(6):
            all_errors[player, :, :] = all_errors[player, :, :] * num_trials[player]      
            for left_block in session.blocks:
                if left_block.choices.shape[0] < 2:
                    continue
                if player >= left_block.getN():
                    continue
                block_errors = np.zeros ([len(win_values), len(decays)])
                for w in range(len(win_values)):
                    for dec in range(len(decays)):
                        vals = np.round(session.values[left_block.item_ids, player], 2)
                        if depth == 1:
                            qvals = POMDP.loadPomdpQVals(left_block.getN(), vals, T_fail, decays[dec], discount, num_round, 6, win_values[w])
                        else:
                            qvals = IPOMDP.loadPomdpQVals(left_block.getN(), vals, T_fail, decays[dec], discount, num_round, 6, depth, win_values[w])
                        block_errors[w, dec] = POMDPFit.findPlayerBlockErrors(left_block, player, qvals, decays[dec], win_values[w], discount, depth)
                other_errors = all_errors[player, :, :] - block_errors
                w, dec = np.unravel_index(other_errors.argmin(), other_errors.shape)
                loocv_error[player] += block_errors[w, dec]  
            #print (player, 1 - loocv_error[player]/ num_trials[player])
        return loocv_error / num_trials    
    
    @staticmethod
    def findPlayerBlockErrors(block, player, qvals, decay_rate, win_value, discount, depth = 1):
        error = 0
        alpha = int(np.sum(block.choices[0, :]))
        beta = int(block.getN() - alpha)
        for step in range(1, block.choices.shape[0]):
            action = np.argmax(qvals[alpha, beta, step, :])
            if (action == 0 and block.choices[step, player] == 1) or (action == 1 and block.choices[step, player] == 0):
                error += 1
            #error += abs(action - block.choices[step, player])
            alpha =  int(alpha * decay_rate)
            beta =  int(beta * decay_rate)
            alpha += int(np.sum(block.choices[step, :]))
            beta += int(block.getN() - np.sum(block.choices[step, :]) )
            if alpha < 1:
                alpha = 1
            if beta < 1:
                beta = 1
        return error
    
    @staticmethod
    def findPlayerEndPredict(block, player, qvals, decay_rate):
        error = 0
        alpha = int(np.sum(block.choices[0, :]))
        beta = int(block.getN() - alpha)
        for step in range(1, block.choices.shape[0]):
            action = np.argmax(qvals[alpha, beta, step, :])
            #if qvals[alpha, beta, step, 0] == qvals[alpha, beta, step, 1]:
            #    action = block.choices[step - 1, player]            
            if action == 1:		    
                if beta == 0:
                    prob_end = 1
                else:
                    prob_end = special.beta(alpha + block.getN()-1, beta) / special.beta(alpha, beta)
            else:
                if alpha == 0:
                    prob_end = 1
                else:
                    prob_end = special.beta(alpha, beta + block.getN() -1) / special.beta(alpha, beta)
	   
            if prob_end > .5:
                error += 1
            
            alpha =  int(alpha * decay_rate)
            beta =  int(beta * decay_rate)
            alpha += int(np.sum(block.choices[step, :]))
            beta += int(block.getN() - np.sum(block.choices[step, :]) )
                
            if alpha < 1:
                alpha = 1
            if beta < 1:
                beta = 1
                
        step_end = block.choices.shape[0]
        action = np.argmax(qvals[alpha, beta, step_end, :])
        if step_end > 1:
            if action == 1:
                if beta == 0:
                    prob_end = 1
                else:
                    prob_end = special.beta(alpha + block.getN()-1, beta) / special.beta(alpha, beta)
            else:
                if alpha == 0:
                    prob_end = 1
                else:
                    prob_end = special.beta(alpha, beta + block.getN() -1) / special.beta(alpha, beta)
            if prob_end <= .5:
                error +=1
        return error

    @staticmethod
    def generateBlockLength(session, num_round = 25, T_fail = .25, discount = .5):
        file = open('pomdp_params1', 'rb')
        all_params = pickle.load(file)
        file = open('pomdp_errors1', 'rb')
        all_errors1 = pickle.load(file)
        all_errors0, all_decays0 = DBN.fitAllPlayers(session, 6, [0,.25, .5, .75, 1], [0])
        all_decays1 = np.zeros(6)
        all_wins1 = np.zeros(6)
        
        qvals = {}
        alphas = {}
        betas = {}
        all_ends = []
        best_level = np.zeros(6)
        for player in range(6):
            all_wins1[player], all_decays1[player] = all_params[(session.id, player)]
            if all_errors1[player] < all_errors0 [player]:
                best_level[player] = 1
                
        for block in session.blocks:
            if block.choices.shape[0] < 2:
                continue
            
            ### load policies and initial values of the block
            for player in range(block.getN()):
                vals = np.round(session.values[block.item_ids, player], 2)
                if best_level[player] == 1:
                    qvals[player] = POMDP.loadPomdpQVals(block.getN(), vals, T_fail, all_decays1[player], discount, num_round, 6, all_wins1[player])
                alphas[player] = int(np.sum(block.choices[0, :]))
                betas[player] = int(block.getN() - alphas[player])
            
            #old_actions = np.array(block.choices[0, :])
            # update for each step
            for step in range(1, 25):
                if step > 9 and np.random.randint(4, size = 1)[0] == 3:
                    break
                actions = np.zeros(block.getN())
                for player in range(block.getN()):
                    random_sample = np.random.uniform(low = 0, high = 1, size = 1)[0]
                    actions[player] = 0
                    if best_level[player] == 0:
                        #actions[player] == int (alphas[player] > betas[player])
                        if random_sample < alphas[player] / float(alphas[player] + betas[player]):
                            actions[player] = 1
                    else:
                        #if random_sample < 1 / (1+ np.exp(100*(qvals[player][alphas[player], betas[player], step, 0] - qvals[player][alphas[player], betas[player], step, 1]))):
                        #    actions[player] = 1
                        actions[player] = int(np.argmax(qvals[player][alphas[player], betas[player], step, :]))
                #if qvals[player][alphas[player], betas[player], step, 0] == qvals[player][alphas[player], betas[player], step, 1]:
                #    actions[player] = old_actions[player]
                if np.unique(actions).size == 1:
                    break
                ## update for the next step              
                #old_actions = np.copy(actions)
                for player in range(block.getN()):
                    if best_level[player] == 0:
                        alphas[player] =  int(alphas[player] * all_decays0[player])
                        betas[player] =  int(betas[player] * all_decays0[player])
                    else:
                        alphas[player] =  int(alphas[player] * all_decays1[player])
                        betas[player] =  int(betas[player] * all_decays1[player])
                    
                    alphas[player] += int(np.sum(actions))
                    betas[player] += int(block.getN() - np.sum(actions))
                    if alphas[player] < 1:
                        alphas[player] = 1
                    if betas[player] < 1:
                        betas[player] = 1
            all_ends.append(step + 1)  
        return all_ends

            
num_round = 25 
discount = .5
T_fail = 0.25
depth = 1

decays = [.25, .5, .75, 1, 0]
decays = [1]
win_values = [0, .1, .2, .5, .75, .9, 1, 1.1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 7.5, 8]
#win_values = [7.5, 6, 3, 1.5, 1.25, 1.1, .9, .5, .2, .1, 0]
win_values = [0, .1, .2, .5, 1, 2, 4, 8]

#all_errors = {}
#all_end_errors = {}
#all_errors_loocv = {}
#all_params = {}
session_list = [s_id for s_id in os.listdir('main/')]

#for i in range(1):
#    session_id = session_list[i]
#    #print (session_id)
#    s = Session(session_id)
#    e, params, ee = POMDPFit.fitAllPlayers(s, decays, win_values, depth, T_fail, discount, num_round)  
#    for player in range(6):
#        all_errors[(session_id, player)] = np.min(e[player, : , :])
#        all_end_errors[(session_id, player)] = ee[player]
#        all_params[(session_id, player)] = params[player]
#        print (player, params[player][0], params[player][1], 1 - all_errors[session_id, player], 1 - ee[player])
#
#pickle.dump (all_errors, open('pomdp_errors' + str(depth), 'wb'))
#pickle.dump (all_errors, open('pomdp_end_errors' + str(depth), 'wb'))
#pickle.dump (all_params, open('pomdp_params' + str(depth), 'wb'))
#
#for i in range(1):
#    session_id = session_list[i]
#    #print (session_id)
#    s = Session(session_id)
#    e = POMDPFit.fitAllPlayersLoocv(s, decays, win_values, depth, T_fail, discount, num_round)     
#    for player in range(6):
#        all_errors_loocv[(session_id, player)] = e[player]  
#        print (player, 1 - e[player])
        
#pickle.dump (all_errors_loocv, open('pomdp_errors_loocv' + str(depth), 'wb'))
#pickle.dump (all_end_errors, open('pomdp_end_errors' + str(depth), 'wb'))

all_ends = np.array([])
for i in range(1):
    session_id = session_list[i]
    session = Session(session_id)
    ends = POMDPFit.generateBlockLength(session, num_round, T_fail, discount)     
    all_ends = np.append(all_ends, ends)
np.save(open("all_ends.npy", 'wb'), all_ends)
print (np.mean(all_ends), all_ends)
