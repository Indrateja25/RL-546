#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:06:32 2022

@author: indra25
"""

import numpy as np
import gym
from gym import spaces
import sys,os
import random
import time
from collections import deque
from datetime import datetime

from agent import *
from plotters import *

def train(no_of_episodes,path):
    
    episodes = no_of_episodes #no of episodes
    gamma = 0.99 #discount-factor    
    epsilon = 1.0 #greedy/random factor
    epsilon_decay_rate = 0.001 #greedy/random factor decay rate
    max_memory_size = 5000 #max replay memory size
    weights_sync_freq = 5 #target network update timesteps 
    mini_batch_size = 128
    model_fit_freq = 10
    weights_ratio = 0.001 #Tau

    #set tracking variables
    epsilon_vals = np.array([epsilon]) #do eps-decay
    total_path_travelled = np.array([]) #for no of tranisitions-made
    total_reward_per_ep = np.array([]) #for total rewards collected in an episode
    random_actions_per_ep = np.array([]) #for no of random actions took per episode
    greedy_actions_per_ep = np.array([]) #for no of greedy actions took per episode
    
    #set environment and agent
    env = gym.make('LunarLander-v2')
    env.reset()
        
    #initializes agent with Q and target-Q
    agent = RandomAgent(env,max_memory_size,epsilon,epsilon_decay_rate)
    agent.reset()


    #start training
    for ep in range(1,episodes+1):
        #now = time.time()
        #print("Episode No:",ep,"time taken till this ep:",np.round(now-program_starts,2),"s")
                        
        #trackers for episodic data
        #start_state = S #start state of the episode
        episode_path = np.array([]) #path of the episode
        episode_rewards = 0 #rewards collected in the episode
        greedy_count = 0 #greedy actions took in the episode
        random_count = 0 #random actions took in the episode
        time_steps = 1
        
        obs = env.reset() #reset env beore each episode
        S = obs[0]
        
        #till it's not a terminal state
        while True:  

            A, action_type = agent.step(S) #agent chooses an action
            
            NS , R, Done, _, Info = env.step(A) #env returns feedback
            R = np.round(R,2)
            
            if time_steps > env._max_episode_steps:
                R = -100
                Done = True

            #print(S,A,NS,R)
            agent.memory.push((S,A,R,NS,Done))

            if time_steps % model_fit_freq == 0 : #& action_type == 1 :
                if agent.memory.check_if_min_samples_exist(mini_batch_size):
                    X,y = agent.replay(mini_batch_size,gamma)
                    agent.fit_Q(X,y,mini_batch_size)
            
            
            episode_path = np.append(episode_path,S) #add to episode path
            episode_rewards += R #increase episode rewards
            
            S = NS #change state to next state
            #env.state = S #change environment's state
            
            #print("time-step:",time_steps)
            #if timesteps are maxed, terminate the episode
            if Done:
                break
            time_steps += 1
            
            env.render()
            
            #increase action counts accordingly
            if action_type == 0: #random
                random_count += 1 
            if action_type == 1: #greedy
                greedy_count += 1

        print("episode: {}/{}, epsilon: {}, time_steps taken: {}, total rewards: {}".format(
                          ep, episodes, np.round(agent.epsilon,2), time_steps-1, episode_rewards))

        if ep % weights_sync_freq == 0 & ep > 0: #update weights every 5 episodes
            theta = agent.model.get_weights()
            theta_1 = agent.target_model.get_weights()

            theta = np.array(theta,dtype='object')
            theta_1 = np.array(theta_1,dtype='object')

            agent.target_model.set_weights = (weights_ratio * theta) + ( (1-weights_ratio) * theta_1)

        #decay epsilon
        agent.epsilon = agent.epsilon*((1-agent.epsilon_decay_rate)**ep) #exponential decay
        agent.epsilon = max(agent.epsilon,0.01) #keep epsilon to a min value so that 
                                                #agent can explore even in latter stages
        episode_rewards = np.round(episode_rewards,2)
        #update all tracking-variables
        epsilon_vals = np.append(epsilon_vals,agent.epsilon) #
        total_reward_per_ep = np.append(total_reward_per_ep,episode_rewards)
        total_path_travelled = np.append(total_path_travelled,len(episode_path))
        random_actions_per_ep = np.append(random_actions_per_ep,random_count)
        greedy_actions_per_ep = np.append(greedy_actions_per_ep,greedy_count)


    #env.render()
    plot_eps_decay(epsilon_vals) #plot epsiolon values over time
    plot_cum_rewards(total_reward_per_ep) #plot total rewards for each episode
    #plot_epsiode_q_vals(q_vals_all_episodes)
    plot_action_types_count(random_actions_per_ep,greedy_actions_per_ep) 
    plot_paths_travelled(total_path_travelled) #plot total transitions for each episode
    #plot_action_types_count(action_per_ep) 
    
    agent.target_model.save(path)
    

def test(episodes,path):
    env = gym.make('LunarLander-v2')
    env.reset()
    
    agent = RandomAgent(env)
    agent.reset()
    model = load_model(path)
    agent.model.set_weights = model.get_weights()
    agent.target_model.set_weights = model.get_weights()
    
    for ep in range(1,episodes+1):
            
            env.reset()
            S = env.observation_space.sample()
            
            episode_rewards = 0
            time_steps = 1
            
            #till it's not a terminal state
            while True:  

                A = agent.test_step(S) #agent chooses an action
                NS , R, Done,_, Info = env.step(A) #env returns feedback

                S = NS #change state to next state
                env.state = S #change environment's state
                
                episode_rewards += R
                
                #if timesteps are maxed, terminate the episode
                if Done:
                    break

                #env.render()
            print("episode: {}/{}, reward: {}, ".format(
                    ep, episodes, episode_rewards))
                    
   
random.seed(1)


path = "lunar_model_ddqn.h5"
print("train results")
t1 = datetime.now()
train(1000,path)
t2 = datetime.now()
print("total training time taken(in minutes): ",np.round((t2-t1).total_seconds()/60,2))


print("test results")
test(10,path)
