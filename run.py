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

rdm_quality = False  # assign item quality randomly
calcPerf = [True, False, False]  # calculate performance (happiness,distance,topK)
plotQuality = False  # plot item quality
plotHistory = False  # plot rating history
fig_idx = 0
num_free = 1
num_runs = 10
num_item = 50
num_user = 10000
items = {}
users = {}
lower, upper = 0,1  # lower and upper bound of item quality
ability_range = range(1, 6)  # ability of 1~5
K = 10  # number of items for performance measurement "top K in expected top K"
rankModes = ['random', 'quality', 'upvotes', 'ucb', 'lcb', 'popularity']
viewModes = ['first', 'position']
viewMode = viewModes[1]
p_pos = 0.5
user_c = 0.5
tau = 1
coeff = 0.5


#********** Initilization
def initialize(seed):

    random.seed(seed)
    np.random.seed(seed)

    #***** Initialization of items
    if not rdm_quality:  # assume item qualities follow a normal distribution between 0~1
        mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        qualities = stats.truncnorm(
            a, b, loc=mu, scale=sigma).rvs(size=num_item)
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
        # each item get free views
        for k in range(num_free):
            items[i].views += 1
            initialEval = user0.evaluate(items[i], method='upvote_only')
            if initialEval:
                items[i].setVotes(initialEval)

    #***** Initialization of users
    for i in range(num_user):
        a = random.choice(ability_range)
        users[i] = User(i, a)

    return items, users


#********** Simulation
def simulate(items, users, rankMode):
    # np.random.seed(123)
    platform = Platform(items=deepcopy(items), users=users)
    perfmeas = platform.run(
        rankMode=rankMode,
        viewMode=viewMode,
        evalMethod="upvote_only",
        perf=calcPerf,
        perfmeasK=K,
        numFree=num_free,
        p_pos=p_pos,
        user_c=user_c,
        tau=tau,
        c=coeff)
    return perfmeas


#********** Start
pool = mp.Pool()
t0 = time.time()
print("-----Start\nnum_runs: {}\nnum_item: {}\nnum_user: {}".format(
    num_runs, num_item, num_user))

# Initialize
seeds = range(num_runs)
inits = pool.map(initialize, seeds)
items, users = zip(*inits)
for i in range(len(seeds)):
    qualities = [itm.getQuality() for itm in items[i].values()]
    mean_quality = np.mean(qualities)
    # print("Seed: {:2}   mean: {}".format(seeds[i], mean_quality))

t_ini = time.time()
print("-----Initialization takes {:.4f}s".format(t_ini - t0))

# Simulate
results = []
for i in range(len(rankModes)):
    result = pool.starmap_async(simulate,
                                zip(items, users, repeat(rankModes[i])))
    results.append(result)

#perfmeas = map(lambda x: x.get(), results)
perfmeas = list(map(lambda x: x.get(), results))

t_done = time.time()
print("-----Simulation takes {:.4f}s".format(t_done - t_ini))

#itms = [
#        list(map(lambda x: [pf['items'] for pf in x], p)) for p in perfmeas
#    ]

#**********  Performance Measurements
if calcPerf[0]:
    happy = [
        list(map(lambda x: [pf['happy'] for pf in x], p)) for p in perfmeas
    ]
    happy = np.mean(np.array(happy), axis=1)
    for i in range(len(rankModes)):
        print("Mode: {:10} happiness: {}".format(rankModes[i], happy[i][-1]))

if calcPerf[1]:
    ktd = [list(map(lambda x: [pf['ktd'] for pf in x], p)) for p in perfmeas]
    ktd = np.mean(np.array(ktd), axis=1)
    for i in range(len(rankModes)):
        print("Mode: {:10} distance: {}".format(rankModes[i], ktd[i][-1]))

if calcPerf[2]:
    topK = [list(map(lambda x: [pf['topK'] for pf in x], p)) for p in perfmeas]
    topK = np.mean(np.array(topK), axis=1)
    for i in range(len(rankModes)):
        print("Mode: {:10} top K percent: {}".format(rankModes[i], topK[i][-1]))

#********** Plotting
if calcPerf[0]:  # user happiness
    fig_idx += 1
    plt.figure(fig_idx)
    for i in range(len(rankModes)):
        plt.plot(happy[i], label='rank by %s' % (rankModes[i]))
    plt.title('user happiness VS. time (c={})'.format(coeff))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('user happiness')
    y_ub = np.ceil(np.max(happy) * 10) / 10
    y_lb = np.floor(np.min(happy) * 10) / 10
    plt.ylim([y_lb, y_ub])
    plt.legend(loc=4)
    plt.grid()
    plt.show()

if calcPerf[1]:  # distance
    fig_idx += 1
    plt.figure(fig_idx)
    for i in range(len(rankModes)):
        plt.plot(ktd[i], label='rank by %s' % (rankModes[i]))
    plt.title('kendall tau distance VS. time (c={})'.format(coeff))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('kendall tau distance')
    y_lb = np.min(ktd)
    y_lb = np.floor(y_lb * 10) / 10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()

if calcPerf[2]:  # top K
    fig_idx += 1
    plt.figure(fig_idx)
    for i in range(len(rankModes)):
        plt.plot(topK[i], label='rank by %s' % (rankModes[i]))
    plt.title("top {} percentage VS. time (c={})".format(K, coeff))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel("top {} percentage".format(K))
    y_lb = np.min(topK)
    y_lb = np.floor(y_lb * 10) / 10
    plt.ylim([y_lb, 1.1])
    plt.legend()
    plt.grid()
    plt.show()

if plotHistory:
    fig_idx += 1
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

# t_plt = time.time()
# print("-----Plotting takes {:.4f}s".format(t_plt - t_done))
