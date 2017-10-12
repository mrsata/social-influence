#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:47:33 2017

@author: xiranliu
"""

from item import Item
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    

def happiness(itms,time,count="both"):
    if count=="upvotes":
        num_upvotes = [itm.getUpVotes() for itm in itms]
    elif count=="downvotes":
        num_upvotes = [-itm.getDownVotes() for itm in itms]
    else:
        num_upvotes = [itm.getUpVotes()-itm.getDownVotes() for itm in itms]
    sum_upvotes = np.sum(num_upvotes)
    upvotes_t = sum_upvotes/time
#    num_views = [itm.getViews() for itm in itms]
#    vote_view_ratio = np.sum(num_upvotes)/np.sum(num_views) 
    return upvotes_t

def printPerfmeas(platform,num_user,K):
    final_list = [itm for itm in platform.items.values()]  # list of items
    print ()
    print ("-----Final Performance after",num_user,"runs-----")
    #***** kendall Tau Distance
    final_ranking = [i + 1 for i in platform.placeOfItems]  # places of items
    ktd = kendallTauDist(final_list, final_placements = final_ranking, rank_std="quality")
    print ()
    print("Kendall tau distance:", ktd['dist'])
    print("Final rankings: ", ktd['final_rank'])
    print("Expected rankings: ", ktd['exp_rank'])
    #***** Top K Percentage
    topK = topKinK(final_list,K=K,final_order = platform.itemRanking, rank_std="random")
    print ()
    print("Percentage of top",K,"items that are actually in top ",K,":",topK['percent'])
    print("Final top",K,"order:", topK['final_order'][:K])
    print("Expected top",K,"order:", topK['exp_order'][:K])
    #***** User Happiness (total #upvotes)
    happy = happiness(final_list,num_user,count="upvotes")
    #happiness = totalNumUpVotes/time
    print ()
    print("User happiness:", happy)


def plotKDT(fig_idx,ktds,rankMode):
    plt.figure(fig_idx)
    plt.plot(ktds,label='rank by %s'%(rankMode))
    plt.title('kendall tau distance VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('kendall tau distance')
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()
    
def plotTopK(fig_idx,topKs,rankMode,K):
    plt.figure(fig_idx)
    plt.plot(topKs,label='rank by %s'%(rankMode))
    plt.title('percentage of top %d in %d VS. time'%(K,K))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('percentage of top %d in %d '%(K,K))
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()
    
def plotHappiness(fig_idx,happy,rankMode):
    plt.figure(fig_idx)
    plt.plot(happy,label='rank by %s'%(rankMode))
    plt.title('user happiness VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('user happiness')
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()

# Plot the evalution history
def plotEvalHistory(fig_idx,platform,evalHistory,num_item):
    plt.figure(fig_idx)
    plt.imshow(evalHistory, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title('Individual Evaluation History')
    plt.minorticks_on()
    plt.xlabel('user')
    plt.ylabel('item')
    plt.tick_params(labelright='on')
    plt.yticks(
        range(0, num_item), [
            str(i) + "(" + str(int(itm.getQuality() * 100) / 100) + ")"
            for i, itm in enumerate(platform.items.values())
        ],
        fontsize=6)
    plt.colorbar(orientation='horizontal')
    plt.show()
    return fig_idx