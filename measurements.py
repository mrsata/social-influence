#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:47:33 2017

@author: xiranliu
"""

from item import Item
import numpy as np
import scipy.stats as stats

# Spearman's Rho
# TODO

# Kendall tau distance
''' Performance measured using Kendall tau distance (bubble sort distance) between final list ranking and expected (quality) ranking
    Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement. 
'''
def kendallTauDist(itms,final_placements = None,rank_std="rdm"):
    itms_final = np.copy(itms) # a copy of the final list
    if rank_std=="random":
        # Reorder the final list in #votes order if ranking strategy is random
#        itms_final = sorted(itms_final, key=lambda x: x.getQuality(), reverse=True)    
        # Ranking in descending upvotes order
        final_rank = stats.rankdata([-itm.getUpVotes() for itm in itms_final],method='min')
    else:  
        final_rank = final_placements  # ranking in displaced order

    # Ranking in descending quality order
    desq_rank = stats.rankdata([-itm.getQuality() for itm in itms_final],method='min')
    # Calculate the Kendall tau distance 
    tau, p_value = stats.kendalltau(desq_rank, final_rank)
    return {'final_rank':final_rank, 'exp_rank':desq_rank ,'dist':tau}

def topKinK(itms,K,final_order = None,rank_std="random"):
    itms_final = np.copy(itms) # a copy of the final list
    exp_order = sorted(range(len(itms_final)),key=lambda x:itms_final[x].getQuality(),reverse=True)
    expected_top_K = exp_order[:K]
    if rank_std=="random":
        # Reorder the final list in #votes order if ranking strategy is random
        final_order = sorted(range(len(itms_final)),key=lambda x:itms_final[x].getUpVotes(),reverse=True)    
    actual_top_K = final_order[:K]
    in_top_K = set(expected_top_K).intersection(actual_top_K)
    topK = len(in_top_K)/K
    return {'percent':topK,'exp_order':exp_order,'final_order':final_order}
    

