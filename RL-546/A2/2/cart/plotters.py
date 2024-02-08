#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:04:24 2022

@author: indra25
"""

import matplotlib.pyplot as plt

def plot_env_current_state(state):    
    plt.figure(figsize=(3,3))
    plt.imshow(state)
    plt.title("Grid-State")
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.show()

def plot_eps_decay(epsilon_vals):
    plt.figure(figsize=(10,6))
    plt.plot(epsilon_vals)
    plt.title('Epsilon-decay')
    plt.ylabel('Epsilon')
    plt.xlabel('Epsiode No')
    plt.show()
    
def plot_cum_rewards(total_reward_per_ep):
    plt.figure(figsize=(10,6))
    plt.plot(total_reward_per_ep+300)
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
    plt.ylabel('No of Actions')
    plt.xlabel('Epsiode No')
    plt.legend(loc='best')
    plt.show()
    