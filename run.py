from copy import deepcopy
from itertools import repeat
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import time

from item import Item
from user import User
from plat2d import Platform
from measurements import *

random.seed(123)
np.random.seed(123)
rdm_quality = False  # assign item quality randomly
plotPerf = True  # plot performance
plotQuality = False  # plot item quality
plotHistory = False  # plot rating history
fig_idx = 0
num_item = 20
num_user = 100000
items = {}
users = {}
lower, upper = 0, 1  # lower and upper bound of item quality
ability_range = range(1, 6)  # ability of 1~5
K = 10  # number of items for performance measurement "top K in expected top K"
rankModes = ['random', 'quality', 'ucb', 'lcb', 'upvotes', 'views']
viewModes = ['first', 'position']
viewMode = viewModes[0]

t0 = time.time()
print("-----Start")

#********** Initilization
#***** Initialization of items
if not rdm_quality:  # assume item qualities follow a normal distribution between 0~1
    mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    qualities = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size=num_item)
    if plotQuality:
        fig_idx += 1
        plt.figure(fig_idx)
        plt.hist(qualities, normed=True)
        plt.title("Distribution of Item Quality")
        plt.show()

user0 = User(0, ability_range[-1])
# assign qualities to items
for i in range(num_item):
    if rdm_quality:
        q = random.uniform(lower, upper)
    else:
        q = qualities[i]
    items[i] = Item(i, q)
    # each item get 5 free views
    for k in range(0, 10):
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
def simulate(args):
    items, users, rankMode = args
    platform = Platform(items=deepcopy(items), users=users)
    perfmeas = platform.run(
        rankMode=rankMode,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perfmeasK=K)
    return perfmeas


pool = mp.Pool(processes=6)
perfmeas = pool.map(simulate, zip(repeat(items), repeat(users), rankModes))

#********** Performance Measurements
# printPerfmeas(platform, num_user, K,rankMode)
# printPerfmeas(platform2, num_user, K,rankMode2)
# printPerfmeas(platform3, num_user, K,rankMode3)
# printPerfmeas(platform4, num_user, K,rankMode4)
# printPerfmeas(platform5, num_user, K,rankMode5)
# printPerfmeas(platform6, num_user, K,rankMode6)

# Timing
t_done = time.time()
print("\n-----Simulation takes", t_done - t_ini)

#***** Trends of performances
happy = list(map(lambda x: [pf['happy'] for pf in x], perfmeas))
# ktds1 = [pf['ktd'] for pf in perfmeas1]
# topKs1 = [pf['topK'] for pf in perfmeas1]
# happy1 = [pf['happy'] for pf in perfmeas1]
# ktds2 = [pf['ktd'] for pf in perfmeas2]
# topKs2 = [pf['topK'] for pf in perfmeas2]
# happy2 = [pf['happy'] for pf in perfmeas2]
# ktds3 = [pf['ktd'] for pf in perfmeas3]
# topKs3 = [pf['topK'] for pf in perfmeas3]
# happy3 = [pf['happy'] for pf in perfmeas3]
# ktds4 = [pf['ktd'] for pf in perfmeas4]
# topKs4 = [pf['topK'] for pf in perfmeas4]
# happy4 = [pf['happy'] for pf in perfmeas4]
# ktds5 = [pf['ktd'] for pf in perfmeas5]
# topKs5 = [pf['topK'] for pf in perfmeas5]
# happy5 = [pf['happy'] for pf in perfmeas5]
# ktds6 = [pf['ktd'] for pf in perfmeas6]
# topKs6 = [pf['topK'] for pf in perfmeas6]
# happy6 = [pf['happy'] for pf in perfmeas6]

L = len(rankModes)
#********** Plotting
if plotPerf:
    # # kendall tau distance
    # fig_idx += 1
    # plt.figure(fig_idx)
    # plotKDT(plt, ktds1, rankMode)
    # plotKDT(plt, ktds2, rankMode2)
    # plotKDT(plt, ktds3, rankMode3)
    # plotKDT(plt, ktds4, rankMode4)
    # plotKDT(plt, ktds5, rankMode5)
    # plotKDT(plt, ktds6, rankMode6)
    # plt.title('kendall tau distance VS. time')
    # plt.minorticks_on()
    # plt.xlabel('time')
    # plt.ylabel('kendall tau distance')
    # y_lb = min(min(ktds1,ktds2,ktds3,ktds4,ktds5,ktds6))
    # y_lb = np.floor(y_lb*10)/10
    # plt.ylim([y_lb, 1.1])
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # # top k in k
    # fig_idx += 1
    # plt.figure(fig_idx)
    # plotTopK(plt, topKs1, rankMode, K)
    # plotTopK(plt, topKs2, rankMode2, K)
    # plotTopK(plt, topKs3, rankMode3, K)
    # plotTopK(plt, topKs4, rankMode4, K)
    # plotTopK(plt, topKs5, rankMode5, K)
    # plotTopK(plt, topKs6, rankMode6, K)
    # plt.title('percentage of top %d in %d VS. time' % (K, K))
    # plt.minorticks_on()
    # plt.xlabel('time')
    # plt.ylabel('percentage of top %d in %d ' % (K, K))
    # y_lb = min(min(topKs1,topKs2,topKs3,topKs4,topKs5,topKs6))
    # y_lb = np.floor(y_lb*10)/10
    # plt.ylim([y_lb, 1.1])
    # plt.legend()
    # plt.grid()
    # plt.show()

    # user happiness
    fig_idx += 1
    plt.figure(fig_idx)
    for i in range(L):
        plotHappiness(plt, happy[i], rankModes[i])
    plt.title('user happiness VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('user happiness')
    y_lb = np.min(np.array(happy))
    y_lb = np.floor(y_lb * 10) / 10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()

if plotHistory:
    fig_idx += 1
    fig_idx = plotEvalHistory(fig_idx, platform, evalHistory, num_item)
