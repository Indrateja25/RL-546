#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:39:48 2022

@author: indra25
"""

import numpy as np
import matplotlib.pyplot as plt
import sys,os
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD,Adam
from keras.models import load_model

class DQN_Model:
    def __init__(self):
        #self.model = self.initialize_DQN_Model()
        pass
        
    #model initialize
    def initialize_DQN_Model(self):
        
        total_states = 16
        total_actions = 4
        
        inputs = keras.Input(shape=(total_states,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(total_actions, activation="linear")(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer = Adam())

        return model

class replay_memory:    
    def __init__(self,max_size):
        self.memory = deque()
        self.max_size = max_size
    
    def push(self,x):
        self.memory.append(x)
        
        self.is_max_size()
        
    def is_max_size(self):
        if len(self.memory) > self.max_size:
            self.memory.popleft()
    
    def sample_transitions(self,n):  
        if n > len(self.memory):
            return []        
        transitions = random.sample(self.memory, n)
        return transitions
    
    def check_if_min_samples_exist(self,n):
        if n > len(self.memory):
            return 0
        return 1

class RandomAgent(DQN_Model, replay_memory):
    
    #initialize the agent
    def __init__(self, env,max_size=3000,epsilon=1,epsilon_decay_rate=0.01):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.terminal_state = env.terminal_state
        self.time_step = env.time_step 
        
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate 
        
        self.model = self.initialize_DQN_Model() #action n/w
        self.target_model = self.initialize_DQN_Model()  #target n/w
        self.target_model.set_weights = self.model.get_weights() #initialize target n/w
        
        self.memory = replay_memory(max_size) #initialize replay memory

        self.action_type = -1 #tracker to track greedy and random actions

        
    #given current state,the agent takes a step to generate an action to perform
    def step(self, state):
        #generate action using epsilon greedy
        rand_num = np.random.random() 
        if self.epsilon > rand_num: # explore
            return np.random.randint(self.action_space.n),0
        else: # exploit, greedy
            action_values = self.predict_q_values(state,"action_q")
            action_index = np.argmax(action_values)
            return action_index,1 # given a state, return action with maximum q-value
        
        return -1
    
    def replay(self,mini_batch_size,gamma):
        
        mini_batch = self.memory.sample_transitions(mini_batch_size)
        
        X = []
        y = []
        for sample in mini_batch:
            state, action, reward, next_state, done = sample[0:5]
            target = reward
            if not done:
                max_Q = np.max(self.predict_q_values(next_state,"target_q")) #target-n/w
                discounted_Q = gamma * max_Q
                target = reward + discounted_Q

            target_f = self.predict_q_values(state,"action_q") #action-n/w 
            target_f[0][action] = target

            state_f = self.preprocess(state)
            X.append(state_f)
            y.append(target_f)

        X = np.array(X)
        X = np.reshape(X,(X.shape[0],16)) 
        y = np.array(y)    
        y = np.reshape(y,(y.shape[0],4))
        
        return X,y
        

    def preprocess(self,state):
        state_f = np.zeros(16)
        state_f[state] = 1
        state_f = np.reshape(state_f, (1,16,))
        return state_f    
    
    def predict_q_values(self,state, model_type):
        state_f = self.preprocess(state)
        action_values = []
        if model_type == 'target_q':
            action_values = self.target_model.predict([state_f],verbose=0)
        if model_type == 'action_q':
            action_values = self.model.predict([state_f],verbose=0)
        return action_values
    
    def fit_Q(self,X,y,mini_batch_size):
        self.model.fit(X,y,batch_size=mini_batch_size,epochs=1,verbose=0)
    
    def test_step(self,state):
        action_values = self.predict_q_values(state,"target_q")
        action_index = np.argmax(action_values)
        return action_index # gien a state, return action with maximum q-value
    
    def print_final_q_values(self):
        #change to find rewards received  and actions taken along the optimal actions across each state.
        for state in range(0,15):
            q_values = self.predict_q_values(state, "target_q")
            optimal_action = np.argmax(q_values)
            print("state: {}, q_values: {}, optimal action: {}".format(state,np.round(q_values,2),optimal_action))

    #resets the agent
    def reset(self):
        pass