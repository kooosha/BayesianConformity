#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 02:27:12 2019

@author: koosha
"""
import numpy as np
import os
from session import Session, Block

class ChooseMax:

    @staticmethod           
    def fitAllPlayers(session):
        num_trials = np.zeros(6)
        error = np.zeros(6)
        for block in session.blocks:
            for step in range(1, block.choices.shape[0]):
                max_choice = 1
                if np.mean(block.choices[step-1, :]) < .5:
                    max_choice = 0
                for player in range(block.getN()):
                    error[player] += abs(max_choice - block.choices[step, player])
                    num_trials[player] +=1
        return error/num_trials

session_list = [s_id for s_id in os.listdir('main/')]
avg = 0
for i in range(0, len(session_list)):
    session_id = session_list[i]
    #print (session_id)
    s = Session(session_id)
    e = ChooseMax.fitAllPlayers(s)
    for p in range(6):
        print (p, 1 - e[p])
        avg += e[p]
         