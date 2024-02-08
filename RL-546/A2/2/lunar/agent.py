#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:02:26 2022

@author: indra25
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import sys,os
import random
import time
from collections import deque
from datetime import datetime


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD,Adam



class DQN_Model:
    def __init__(self,learning_rate):
        #self.model = self.initialize_DQN_Model()
        pass
        
    #model initialize
    def initialize_DQN_Model(self):
        
        inputs = keras.Input(shape=(8,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(4, activation="linear")(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer = Adam(0.001))

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
        n = min(n,len(self.memory)) #sample always only till max size of the replay memory
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
        
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate 
        
        self.model = self.initialize_DQN_Model()
        self.target_model = self.initialize_DQN_Model()
        self.target_model.set_weights = self.model.get_weights()
        
        self.memory = replay_memory(max_size)
        self.time_step = 0 
        
        
    #given current state,the agent takes a step to generate an action to perform
    def step(self, state):
        #generate action using epsilon greedy
        rand_num = np.random.random() 
        if self.epsilon > rand_num: # explore
            return np.random.randint(self.action_space.n),0
        else: # exploit
            action_values = self.predict_q_values(state,"action_q")
            action_index = np.argmax(action_values)
            return action_index,1 # gien a state, return action with maximum q-value
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
                target = reward + gamma * max_Q

            target_f = self.predict_q_values(state,"action_q") #action-n/w 
            target_f[0][action] = target

            state_f = self.preprocess(state)
            X.append(state_f)
            y.append(target_f)

        X = np.array(X)
        X = np.reshape(X,(X.shape[0],8)) 
        y = np.array(y)    
        y = np.reshape(y,(y.shape[0],4))

        return X,y
        
    def preprocess(self,state):
        state_f = np.reshape(state, (1,8))
        return state_f
    
    def predict_q_values(self,state, model_type='target_q'):
        action_values = []
        state_f = self.preprocess(state)
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

    #resets the agent
    def reset(self):
        pass
    
    