#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:17:30 2019

@author: koosha
"""

import numpy as np
from scipy import special,misc
from pomdp import POMDP

class IPOMDP:
    def __init__(self, num_players, values, T_fail = 0, decay_rate = 1, discount_factor = .5, num_round = 25, search_max = 120, depth = 2, win_value = 0):
        self.decay = decay_rate
        self.discount = discount_factor
        self.N = num_players
        self.values = values
        self.depth = depth
        self.win_value = win_value
        alpha_max = search_max +  num_round * (self.N + 1)
        beta_max = search_max + num_round * (self.N + 1)
        self.T = np.zeros([num_round - 1, alpha_max, beta_max, 2])
        self.V = np.zeros([alpha_max, beta_max, num_round])  ## Value function (maximum expected reward)
        self.Q = np.zeros([alpha_max, beta_max, num_round, 2])  ## maximum Expected reward of each action
        self.P = np.zeros([alpha_max, beta_max, num_round])  ## Policy function (best aciton)
        if depth == 2:
            self.preQ1 = POMDP.loadPomdpQVals(num_players, [1.5, 2.5], T_fail, decay_rate, discount_factor, num_round, search_max, win_value)
            self.preQ0 = POMDP.loadPomdpQVals(num_players, [2.5, 1.5], T_fail, decay_rate, discount_factor, num_round, search_max, win_value)
        else:
           self.preQ1 = IPOMDP.loadPomdpQVals(num_players, [1.5, 2.5], T_fail, decay_rate, discount_factor, num_round, search_max, depth - 1, win_value)
           self.preQ0 = IPOMDP.loadPomdpQVals(num_players, [2.5, 1.5], T_fail, decay_rate, discount_factor, num_round, search_max, depth - 1, win_value)        
        self.findAllValues()

    def instantR(self, action, other_actions):
        if action == 0 and other_actions == 0:
            return self.values[0]
        if action == 1 and other_actions == self.N - 1:
            return self.values[1]
        return 0

    def calcValueAction(self, alpha, beta, step):
        num_choice1 = round((self.N-1) * alpha/(alpha+beta))
        num_choice0 = self.N-1 - num_choice1
        other_actions = int(num_choice1 * np.argmax(self.preQ1[alpha, beta, step]) + num_choice0 * np.argmax(self.preQ0[alpha, beta, step]))
        r_0 = self.instantR(0, other_actions)
        r_1 = self.instantR(1, other_actions)
        alpha = int(alpha * self.decay)
        beta = int(beta * self.decay)
        if other_actions > 0:
            r_0 += self.discount * self.V[alpha +  other_actions, 1 + beta + self.N -1 - other_actions, step + 1]
        if other_actions < self.N - 1:
            r_1 += self.discount * self.V[1 + alpha + other_actions, beta + self.N -1 - other_actions, step + 1]
        
        self.V[alpha, beta, step] = r_0
        
        self.Q[alpha, beta, step, 0] = r_0
        self.Q[alpha, beta, step, 0] = r_1
        
        if r_1 > r_0:
            self.P[alpha, beta,  step] = 1
            self.V[alpha, beta,step] = r_1


    def findAllValues(self):
        
        alpha_max , beta_max, num_round = self.V.shape
        # last round
        for alpha in range(alpha_max):
            for beta in range(beta_max):
                if alpha + beta == 0:
                    continue
                num_choice1 = round((self.N-1) * alpha/(alpha+beta))
                num_choice0 = self.N-1 - num_choice1
                other_actions = int(num_choice1 * np.argmax(self.preQ1[alpha, beta, num_round -1]) + num_choice0 * np.argmax(self.preQ0[alpha, beta, num_round-1]))
                self.V[alpha, beta, num_round - 1] = self.instantR(0, other_actions)
                r_1 = self.instantR(1, other_actions)
                self.Q[alpha, beta, num_round - 1, 0] = self.V[alpha, beta, num_round - 1]
                self.Q[alpha, beta, num_round - 1, 1] = r_1
                if r_1 > self.V[alpha, beta, num_round - 1]:
                    self.V[alpha, beta, num_round - 1] = r_1
                    self.P[alpha, beta, num_round - 1] = 1
        # other rounds by dynamic programming:
        for step in range(num_round - 2, -1, -1):
            for alpha in range(alpha_max - (num_round - 1 - step) * self.N):
                for beta in range(beta_max - (num_round - 1 - step) * self.N):
                    if alpha + beta == 0:
                        continue
                    self.calcValueAction(alpha, beta, step)
                    

    @staticmethod
    def loadPomdpQVals(num_players, values, T_fail, decay_rate = 1, discount_factor = 1, num_round = 15, search_max = 100, depth = 2, win_value = 0):
        file_name = 'saved_Qs/I%d_%d_%d_%d_%d_%d_%d_%d_%d.npy' % (depth, num_players, (win_value + values[0]) * 100, values[1] * 100, T_fail* 100, decay_rate * 100, num_round, search_max, discount_factor * 100)
        file_name2 = 'saved_Qs/I%d_%d_%d_%d_%d_%d_%d_%d_%d.npy' % (depth, num_players, (win_value + values[1]) * 100 , values[0] * 100, T_fail* 100, decay_rate * 100, num_round, search_max, discount_factor * 100)
        try:
            return np.load(file_name)
        except:
            try:
                policy = np.load(file_name2)     
                return 1- policy
            except:
                pomdp = IPOMDP(num_players, values, T_fail, decay_rate, discount_factor, num_round, search_max, depth, win_value)
                np.save(file_name, pomdp.Q)
                return pomdp.Q
