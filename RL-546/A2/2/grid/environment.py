#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:38:22 2022

@author: indra25
"""
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


def plot_env_current_state(state):    
    plt.figure(figsize=(3,3))
    plt.imshow(state)
    plt.title("Grid-State")
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.show()

    
class GridEnvironment(gym.Env):
    
    # Initializes the class
    def __init__(self):   
        
        self.observation_space = spaces.Discrete(16) #create observation-space
        self.action_space = spaces.Discrete(4) #create action-space
        self.grid_size = 4 #create grid-size variable
        self.max_time_steps = 50 #set max possible transitions in an episode
        self.time_step = 1 #create timestep-counter
        
        self.rewards = self.reset_rewards() #create/reset rewards for the env
        self.state = 0 #set inital state to zero
        self.terminal_state = 15 #set terminal/goal state
        self.max_reward = np.max(self.rewards)
        
    #takes 1 step forward in the env
    def step(self,action):
        
        next_state = self.get_next_state(action)  #get next-state for the current state,action pair
        
        reward = self.rewards[next_state] #get rewards for going the next-state
        if self.state == next_state: #feedback reward of -20 if agent makes a null move
            reward += -10
                   
        self.rewards[next_state] = -1 #remove rewards and make it just an additional move 
        self.time_step += 1 #increase time-step
        done= False
        #check if max time steps are exceeded or terminal state is reached
        if self.time_step > self.max_time_steps  or next_state == self.terminal_state:
            done = True  
        info = {}
        return next_state, reward, done, info 
    
    #resets the env. 
    def reset(self):
        self.rewards = self.reset_rewards() #create/reset rewards for the env
        self.state = 0 #set inital state to zero
        self.terminal_state = 15 #set terminal/goal state
        self.time_step = 1 #increase timestep
        pass
    
    #render current state
    def render(self):
        state = np.zeros((4,4))
        current_cell = self.state
        current_x,current_y = int(current_cell/4),int(current_cell%4)
        terminal_cell = self.terminal_state
        terminal_x,terminal_y = int(terminal_cell/4),int(terminal_cell%4)
        state[current_x,current_y] = 0.5
        state[terminal_x,terminal_y] = 1
        
        plot_env_current_state(state) #plot current state matrix of the env as an image
    
    #helper, resets rewards
    def reset_rewards(self):
        rewards = np.full((16),-1) 
        rewards[3] = 25
        #rewards[10] = -10
        rewards[12] = -20
        #rewards[13] = 10
        rewards[15] = 100
        
        return rewards
    
    #helper, gets next state given current state and action
    def get_next_state(self,action):
        x = int(self.state / self.grid_size)
        y = int(self.state % self.grid_size )

        if action == 0: #move left
            y -= 1
        elif action == 1: #move right
            y += 1
        elif action == 2: #move up
            x -= 1
        elif action == 3: #move down
            x += 1
        else:
            return -1
 
        x = np.clip(x,0,self.grid_size-1)
        y = np.clip(y,0,self.grid_size-1)
        next_state = (self.grid_size*x) + y
        
        return next_state
        