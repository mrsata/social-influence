#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:47:33 2017

@author: xiranliu
"""

from item import Item
import numpy as np
from scipy.stats import rankdata
from scipy.stats import kendalltau

# Spearman's Rho

# Performance measured using Kendall tau distance (bubble sort distance) between final list ranking and expected (quality) ranking
# Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement. 
def kendallTauDist(itms,final_ranking = None,rank_std="random"):
    itms_final = np.copy(itms) # not to modify the original list accidentally
    print ("Ranking Strategy: ",rank_std)
    if rank_std=="random":
        print ("(For 'random', performance measurement is computed after ranking final list by #Votes.)")
        # Reorder the final list in descending quality order
        itms_final = sorted(itms_final, key=lambda x: x.getQuality(), reverse=True)    
        # Ranking in descending upvotes order
        final_rank = rankdata([-itm.getVotes() for itm in itms_final],method='min')
    elif rank_std=="quality":  
        # Ranking in displaced order
        final_rank = final_ranking
    # Ranking in descending quality order
    desq_rank = rankdata([-itm.getQuality() for itm in itms_final],method='min')
    # Calculate the Kendall tau distance 
    tau, p_value = kendalltau(desq_rank, final_rank)
    return {'final_rank':final_rank, 'exp_rank':desq_rank ,'dist':tau}