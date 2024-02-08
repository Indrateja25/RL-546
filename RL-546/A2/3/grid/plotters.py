#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:41:19 2022

@author: indra25
"""

import matplotlib.pyplot as plt


def plot_eps_decay(epsilon_vals):
    plt.figure(figsize=(10,6))
    plt.plot(epsilon_vals)
    plt.title('Epsilon-decay')
    plt.ylabel('Epsilon')
    plt.xlabel('Epsiode No')
    plt.show()
    
def plot_cum_rewards(total_reward_per_ep):
    plt.figure(figsize=(10,6))
    plt.plot(total_reward_per_ep)
    plt.title('Total Reward Per Episode')
    plt.ylabel('Total Reward')
    plt.xlabel('Epsiode No')
    plt.show()
    
    
def plot_paths_travelled(total_path_travelled):
    plt.figure(figsize=(10,6))
    plt.plot(total_path_travelled)
    plt.title('Total Path Travelled Per Episode')
    plt.ylabel('Steps made in Episode')
    plt.xlabel('Epsiode No')
    plt.show()
    

def plot_action_types_count(random_actions_per_ep,greedy_actions_per_ep):
    plt.figure(figsize=(10,6))
    plt.plot(random_actions_per_ep, label = "random_actions_per_ep")
    plt.plot(greedy_actions_per_ep, label = "greedy_actions_per_ep")
    plt.title('Random vs Greedy actions took per EP')
    plt.ylabel('Total Q-Value')
    plt.xlabel('Epsiode No')
    plt.legend(loc='best')
    plt.show()

def plot_epsiode_q_vals(total_reward_all_episodes):
    n = len(total_reward_all_episodes)
    plt.figure(figsize=(10,6))
    for i in range(0,n):
        plt.plot(total_reward_all_episodes[i],'--o',label="state: "+str(i))
    plt.title('Rewards overtime for every state')
    plt.ylabel('Rewards for episodes starting with random start state')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plot_test_rewards(total_reward_per_ep):
    plt.figure(figsize=(10,6))
    plt.plot(total_reward_per_ep)
    plt.title('Total Reward Per Episode : testing')
    plt.ylabel('Total Reward')
    plt.xlabel('Epsiode No')
    plt.show()