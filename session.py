#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 01:10:47 2019

@author: koosha
"""
import numpy as np
import scipy.io as io

class Block:
    def __init__(self, directory , block_id):
        block_mat = io.loadmat(directory + 'cdm_id1'  + '_b' + str(block_id) + '.mat')
        self.item_ids = block_mat['store_items'][0] - 1
        self.choices = np.array(block_mat['store_o_choices'] - 1, dtype = 'int16')
        self.id = block_id
    def getN(self):
        return self.choices.shape[1]

class Session:
    def __init__(self, session_id):
        directory = 'main/' + str(session_id) +'/'
        self.id = session_id
        self.blocks = []
        self.values = np.zeros([40, 6])
        for player in range(6):
            value_mat = io.loadmat(directory + 'bdm_id' + str(player+1) + '.mat')
            self.values [:, player] = value_mat['valueBDM_data'][:,1]  ### TODO: Try without
        for b in range(1, 41, 1):
            block = Block(directory, b)
            if block.choices.shape[0] > 1:
                self.blocks.append(block)
 

