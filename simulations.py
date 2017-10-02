#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:11:15 2017

@author: xiranliu

Description: Run multiple simulations of specified number of items, number of users, number of simulations, ranking strategies and evaluation methods.
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from item import Item
from user import User
from plat import Platform
from measurements import *

# Simulation
def simulation(num_simulation,num_item,num_user,mode,evalMethod):
    rdm_quality = False
    lower, upper = 0, 1 # lower and upper bound of item quality
    ability_range = range(1, 6)  # ability of 1~5
    
    if not rdm_quality: # assume item qualities follow a normal distribution between 0~1
        mu, sigma = 0.5, 0.3 # mean and standard deviation of item quality
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        qualities = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size=num_item)
                
    # Initialization of users
    users = {}
    for i in range(num_user):
        a = random.choice(ability_range)
        users[i] = User(i, a)
    
    dist = list()
    # Simulations start
    for sim in range(0,num_simulation):
#        print ("Simulation %d" % (sim)) 
#        random.seed(i)
        items = {}
        # Initialization of items
        for i in range(num_item):
            if rdm_quality:
                q = random.uniform(lower, upper)
            else:
                q = qualities[i]
            items[i] = Item(i, q)
        
        # Run simulation once
        platform = Platform(items=items, users=users)
        platform.rankItems(mode=mode)
        platform.placeItems(mode='all')
        # Currently available mode: 'random', 'quality', 'views', 'upvotes', 'ucb'
        # Currently available evalMethod: 'abs_quality', 'rel_quality', 'upvote_only'
        viewHistory, evalHistory = platform.run(mode=mode,evalMethod=evalMethod,run_mode='all')
        
        # Measure the performance
        final_list = [itm for itm in platform.items.values()] # list of items
        ktds = kendallTauDist(final_list)
        dist.append(ktds['dist'])
    return dist


# Run Simulation
num_item = 20
num_user = 50
num_simulation = 10000

evalMethod='upvote_only'

for mode in ['random', 'quality']: # 'views', 'upvotes', 'ucb'
    dist = simulation(num_simulation,num_item,num_user,mode,evalMethod)
    print ("num_item: {}\nnum_user: {}\nnum_simulation: {}".format(num_item, num_user, num_simulation))
    print ("ranking mode: {}\nevaluation method: {}".format(mode, evalMethod))
    print ('Performance (Kendall tau distance):',np.mean(dist))
    print ("")
        
