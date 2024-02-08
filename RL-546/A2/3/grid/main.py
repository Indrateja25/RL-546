#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:37:22 2022

@author: indra25
"""

import numpy as np
import sys,os
import random
from collections import deque
from datetime import datetime

import pandas as pd
from environment import *
from agent import *
from plotters import *

def target_preprocess(y, n):
    if n > 1:
        return np.round(y/n,3)
    return y

def train(no_of_episodes,path):
    
    #alpha = 0.15 #learning-rate
    episodes = no_of_episodes #no of episodes
    gamma = 0.99 #discount-factor    
    epsilon = 1 #greedy/random factor
    epsilon_decay_rate = 0.001 #greedy/random factor decay rate
    
    max_memory_size = 3000 #max replay memory size
    mini_batch_size = 128 #repplay mem, mini batch size
    weights_sync_freq = 5 #target network update freq (episode)
    model_fit_freq = 10 # model.fit freq (time_step)
    weights_ratio = 0.001 #Tau
    
    random_start_state = 0 #is random start state required
    specific_start_state = 0 # not called when random_start_state = 1
    #max_timesteps = 50    

    #set tracking variables
    epsilon_vals = np.array([epsilon]) #do eps-decay
    total_path_travelled = np.array([]) #for no of tranisitions-made
    total_reward_per_ep = np.array([]) #for total rewards collected in an episode
    random_actions_per_ep = np.array([]) #for no of random actions took per episode
    greedy_actions_per_ep = np.array([]) #for no of greedy actions took per episode
    total_reward_all_episodes = [[] for _ in range(16)]
    
    #set environment and agent
    env = GridEnvironment()
    env.reset()
    max_reward = env.max_reward
    
    #initializes agent with Q and target-Q
    agent = RandomAgent(env,max_memory_size,epsilon,epsilon_decay_rate)
    agent.reset()

    try:
        #start training
        for ep in range(1,episodes+1):
           
            env.reset() #reset env beore each episode
            
            #check if we need a random start state or a specified start state for each epiode
            #if random_start_state:
                #choose a random start state
            S = env.observation_space.sample()
            if S == env.terminal_state: # if inital state is final state, terminate the episode
                ep -= 1
                continue
            if not random_start_state:
                S = specific_start_state
            
            env.state = S #set env state to the chosen state

            #trackers for episodic data
            start_state = S #start state of the episode
            episode_path = np.array([]) #path of the episode
            episode_rewards = 0 #rewards collected in the episode
            greedy_count = 0 #greedy actions took in the episode
            random_count = 0 #random actions took in the episode
            
            #till it's not a terminal state
            time_steps = 1
            while S != env.terminal_state:  
                
                A, action_type = agent.step(S) #agent chooses an action
                NS , R, Done, Info = env.step(A) #env returns feedback
                #print(S,A,NS,R)
                
                #R_1 = target_preprocess(R,max_reward) #normalize reward 
                agent.memory.push((S,A,R,NS,Done)) #push to replay memory

                #fit action n/w by sampling transitions from replay memory
                if time_steps % model_fit_freq == 0 : #& action_type == 1 :
                    if agent.memory.check_if_min_samples_exist(mini_batch_size):
                        X,y = agent.replay(mini_batch_size,gamma) 
                        agent.fit_Q(X,y,mini_batch_size)
                
                time_steps += 1

                episode_path = np.append(episode_path,S) #add to episode path
                episode_rewards += R #increase episode rewards

                S = NS #change state to next state
                env.state = S #change environment's state

                #increase action counts accordingly
                if agent.action_type == 0: #random
                    random_count += 1 
                if agent.action_type == 1: #greedy
                    greedy_count += 1

                #if timesteps are maxed, terminate the episode
                if Done:
                    break

                #env.render()
            
            print("episode: {}/{}, start_state: {}, epsilon: {}, time_steps taken: {}, total rewards: {}".format(
                    ep, episodes,start_state, np.round(agent.epsilon,2), time_steps-1, episode_rewards))
            
            
            if ep % weights_sync_freq == 0: #update weights every 5 episodes
                theta = agent.model.get_weights()
                theta_1 = agent.target_model.get_weights()
                
                theta = np.array(theta,dtype='object')
                theta_1 = np.array(theta_1,dtype='object')
                
                agent.target_model.set_weights = (weights_ratio * theta) + ( (1-weights_ratio) * theta_1)

            #decay epsilon
            agent.epsilon = agent.epsilon*((1-agent.epsilon_decay_rate)**ep) #exponential decay
            agent.epsilon = max(agent.epsilon,0.01) #keep epsilon to a min value so that 
                                                    #agent can explore even in latter stages

            #update all tracking-variables
            epsilon_vals = np.append(epsilon_vals,agent.epsilon) #
            total_reward_per_ep = np.append(total_reward_per_ep,episode_rewards)
            total_path_travelled = np.append(total_path_travelled,len(episode_path))
            random_actions_per_ep = np.append(random_actions_per_ep,random_count)
            greedy_actions_per_ep = np.append(greedy_actions_per_ep,greedy_count)
            total_reward_all_episodes[start_state].append(episode_rewards)
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e,exc_type, fname, exc_tb.tb_lineno)


    
    #env.render() 
    plot_eps_decay(epsilon_vals) #plot epsiolon values over time
    plot_cum_rewards(total_reward_per_ep) #plot total rewards for each episode
    plot_epsiode_q_vals(total_reward_all_episodes)
    plot_paths_travelled(total_path_travelled) #plot total transitions for each episode
    plot_action_types_count(random_actions_per_ep,greedy_actions_per_ep) 
    
    agent.target_model.save(path)
    

def test(episodes,path):
    env = GridEnvironment()
    env.reset()
    
    agent = RandomAgent(env)
    agent.reset()
    #agent.target_model.set_weights = model.get_weights() 
    agent.target_model = load_model(path)
    
    total_rewards = []
    
    for ep in range(1,episodes+1):
        
            env.reset()
            S = env.observation_space.sample()
            if S == env.terminal_state: # if inital state is final state, terminate the episode
                ep -= 1
                continue
            env.state = S #set env state to the chosen state
            
            episode_rewards = 0
            path_taken = [S]
            time_steps = 1
            
            #till it's not a terminal state
            while S != env.terminal_state:  

                A = agent.test_step(S) #agent chooses an action
                NS , R, Done, Info = env.step(A) #env returns feedback
                #print(S,A,NS,R, Done)
                
                S = NS #change state to next state
                env.state = S #change environment's state
                
                time_steps += 1
                episode_rewards += R
                path_taken.append(S)
                
                #if timesteps are maxed, terminate the episode
                if Done:
                    break
                
                #env.render()
             
            print("episode: {}/{}, reward: {}, time_steps:{} ".format(
                    ep, episodes, episode_rewards,time_steps))
            
            total_rewards.append(episode_rewards)
    plot_test_rewards(total_rewards)
    agent.print_final_q_values()
    
random.seed(1)

path = "grid_model_ddqn.h5"
print("train results")
t1 = datetime.now()
train(300,path)
t2 = datetime.now()
print("total training time taken(in minutes): ",np.round((t2-t1).total_seconds()/60,2))


print("test results")
test(10,path)






