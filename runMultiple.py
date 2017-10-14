from copy import deepcopy
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import time

from item import Item
from user import User
from plat import Platform
from measurements import *

t0 = time.time()
print("-----Start")
parser = argparse.ArgumentParser()
parser.add_argument('--rankMode', type=str, default='upvotes')
parser.add_argument('--placeMode', type=str, default='all')
parser.add_argument('--viewMode', type=str, default='first')
args = parser.parse_args()
rdm_quality = False  # assign item quality randomly
plotPerf = True  # plot performance

numRealizations = 20
ktds = list()
topKs = list()
happy = list()

for realization in range(0,numRealizations):
    random.seed(123)
    np.random.seed(123)
    fig_idx = 0
    num_item = 15
    num_user = 100000
    items = {}
    users = {}
    lower, upper = 0, 1  # lower and upper bound of item quality
    ability_range = range(1, 6)  # ability of 1~5
    K = 10  # number of items for performance measurement "top K in expected top K"
    # rankModes = ['', 'random', 'quality', 'views', 'upvotes', 'lcb', 'ucb']
    rankModes = ['', 'random', 'quality', 'ucb', 'lcb', 'upvotes', 'views']
    rankMode = args.rankMode
    rankMode2 = rankModes[2]
    rankMode3 = rankModes[3]
    rankMode4 = rankModes[4]
    rankMode5 = rankModes[5]
    rankMode6 = rankModes[6]
    
    viewModes = ['first', 'position']
    viewMode = viewModes[0]
    
    #********** Initilization
    #***** Initialization of items
    if not rdm_quality:  # assume item qualities follow a normal distribution between 0~1
        mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        qualities = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size=num_item)
            
    user0 = User(0, ability_range[-1])
    # assign qualities to items
    for i in range(num_item):
        if rdm_quality:
            q = random.uniform(lower, upper)
        else:
            q = qualities[i]
        items[i] = Item(i, q)
        # each item get 5 free views
        for k in range(0,10):
            initialEval = user0.evaluate(items[i], method='upvote_only')
            if initialEval:
                items[i].setVotes(initialEval) 
    # quality statistics
    qualities = [itm.getQuality() for itm in items.values()]
    mean_quality = np.mean(qualities)
    min_quality = min(qualities)
    max_quality = max(qualities)
    print("Item Qualities: \n mean {} \n min  {} \n max  {}".format(
        mean_quality, min_quality, max_quality))
    
    
    #***** Initialization of users
    for i in range(num_user):
        a = random.choice(ability_range)
        users[i] = User(i, a)
    
    print("num_item: {}, num_user: {}".format(num_item, num_user))
    
    t_ini = time.time()
    print("-----Initialization takes", t_ini - t0)
    
    #********** Run simulation once
    items2 = deepcopy(items)
    items3 = deepcopy(items)
    items4 = deepcopy(items)
    items5 = deepcopy(items)
    items6 = deepcopy(items)
    
    platform = Platform(items=items, users=users)
    viewHistory, evalHistory = platform.run(
        rankMode=rankMode,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals = np.cumsum(evalHistory, axis=1)  # accumulation of evaluations
    
    platform2 = Platform(items=items2, users=users)
    viewHistory2, evalHistory2 = platform2.run(
        rankMode=rankMode2,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals2 = np.cumsum(evalHistory2, axis=1)
    
    platform3 = Platform(items=items3, users=users)
    viewHistory3, evalHistory3 = platform3.run(
        rankMode=rankMode3,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals3 = np.cumsum(evalHistory3, axis=1)
    
    platform4 = Platform(items=items4, users=users)
    viewHistory4, evalHistory4 = platform4.run(
        rankMode=rankMode4,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals4 = np.cumsum(evalHistory4, axis=1)
    
    platform5 = Platform(items=items5, users=users)
    viewHistory5, evalHistory5 = platform5.run(
        rankMode=rankMode5,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals5 = np.cumsum(evalHistory5, axis=1)
    
    platform6 = Platform(items=items6, users=users)
    viewHistory6, evalHistory6 = platform6.run(
        rankMode=rankMode6,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    accum_evals6 = np.cumsum(evalHistory6, axis=1)
    
    #********** Performance Measurements
    printPerfmeas(platform, num_user, K,rankMode)
    printPerfmeas(platform2, num_user, K,rankMode2)
    printPerfmeas(platform3, num_user, K,rankMode3)
    printPerfmeas(platform4, num_user, K,rankMode4)
    printPerfmeas(platform5, num_user, K,rankMode5)
    printPerfmeas(platform6, num_user, K,rankMode6)
    
    # Timing
    print()
    t_done = time.time()
    print("-----Simulation takes", t_done - t_ini)
    
    #***** Trends of performances
    perfmeas1 = platform.perfmeas
    ktds1 = [pf['ktd'] for pf in perfmeas1]
    topKs1 = [pf['topK'] for pf in perfmeas1]
    happy1 = [pf['happy'] for pf in perfmeas1]
    perfmeas2 = platform2.perfmeas
    ktds2 = [pf['ktd'] for pf in perfmeas2]
    topKs2 = [pf['topK'] for pf in perfmeas2]
    happy2 = [pf['happy'] for pf in perfmeas2]
    perfmeas3 = platform3.perfmeas
    ktds3 = [pf['ktd'] for pf in perfmeas3]
    topKs3 = [pf['topK'] for pf in perfmeas3]
    happy3 = [pf['happy'] for pf in perfmeas3]
    perfmeas4 = platform4.perfmeas
    ktds4 = [pf['ktd'] for pf in perfmeas4]
    topKs4 = [pf['topK'] for pf in perfmeas4]
    happy4 = [pf['happy'] for pf in perfmeas4]
    perfmeas5 = platform5.perfmeas
    ktds5 = [pf['ktd'] for pf in perfmeas5]
    topKs5 = [pf['topK'] for pf in perfmeas5]
    happy5 = [pf['happy'] for pf in perfmeas5]
    perfmeas6 = platform6.perfmeas
    ktds6 = [pf['ktd'] for pf in perfmeas6]
    topKs6 = [pf['topK'] for pf in perfmeas6]
    happy6 = [pf['happy'] for pf in perfmeas6]
    
    ktds.append([ktds1,ktds2,ktds3,ktds4,ktds5,ktds6])
    topKs.append([topKs1,topKs2,topKs3,topKs4,topKs5,topKs6])
    happy.append([happy1,happy2,happy3,happy4,happy5,happy6])

# Average across realizations
ktds = np.mean(np.array(ktds),axis=0)
topKs = np.mean(np.array(topKs),axis=0)
happy = np.mean(np.array(happy),axis=0)
rankModes = [rankMode,rankMode2,rankMode3,rankMode4,rankMode5,rankMode6]
L = len(ktds)

#********** Plotting
if plotPerf:
    # kendall tau distance
    fig_idx += 1
    plt.figure(fig_idx)
    for l in range(0,L):
        plotKDT(plt, ktds[l,:], rankModes[l])
    plt.title('kendall tau distance VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('kendall tau distance')
    y_lb = np.min(ktds)
    y_lb = np.floor(y_lb*10)/10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()

    # top k in k
    fig_idx += 1
    plt.figure(fig_idx)
    for l in range(0,L):
        plotTopK(plt, topKs[l,:], rankModes[l],K)
    plt.title('percentage of top %d in %d VS. time' % (K, K))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('percentage of top %d in %d ' % (K, K))
    y_lb = np.min(topKs)
    y_lb = np.floor(y_lb*10)/10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()

    # user happiness
    fig_idx += 1
    plt.figure(fig_idx)
    for l in range(0,L):
        plotHappiness(plt, happy[l,:], rankModes[l])
    plt.title('user happiness VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('user happiness')
    y_lb = np.min(happy)
    y_lb = np.floor(y_lb*10)/10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()
