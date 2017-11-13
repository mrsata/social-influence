from __future__ import division
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
plotQuality = False  # plot item quality
num_free = 1  # number of free views upon initialization
num_runs = 10  # number of realizations
num_item = 50  # total number of items
num_user = 100000  # total number of users (time)
lower, upper = 0, 1  # lower and upper bound of item quality
mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
ability_range = range(1, 6)  # ability of 1~5
rankModes = ['quality', 'ucb']
viewModes = ['first', 'position']
viewMode = viewModes[1]  # how platform displays items to users
n_showed = 50  # number of items displayed by the platform
p_pos = 0.5  # ratio of positional preference in user's choice
# p_pos=1 has only positional prefernece
user_c = 0.5  # coeff of user's lcb
tau_and_cs = np.mgrid[0:3.1:.1, 0:3.1:.1]
tau_and_cs = np.concatenate(
    (tau_and_cs[0][:, :, np.newaxis], tau_and_cs[1][:, :, np.newaxis]),
    axis=2).reshape(-1, 2)
tol_cnvrg = 0.005
tol_close = 0.005
num_consec = 1000
# tau_and_cs = np.array(([1, .5], ))


#********** Initilization
def initialize(seed):

    random.seed(seed)
    np.random.seed(seed)
    items = {}
    users = {}

    #***** Initialization of items
    if not rdm_quality:  # assume item qualities follow a normal distribution between 0~1
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
def simulate(inputs):
    # np.random.seed(123)
    items, users, tau_and_c = inputs
    tau, c = tau_and_c
    platform1 = Platform(items=deepcopy(items), users=users)
    platform2 = Platform(items=deepcopy(items), users=users)
    happys1 = np.zeros(num_consec)
    happys2 = np.zeros(num_consec)
    step = 0
    while step < num_user:
        # if step % 500 == 0:
        # print(step)
        happys0 = happys1
        happy1 = platform1.step(
            uid=step,
            rankMode=rankModes[0],
            viewMode=viewMode,
            evalMethod="upvote_only",
            numFree=num_free,
            n_showed=n_showed,
            p_pos=p_pos,
            user_c=user_c,
            tau=tau,
            c=c)
        happy2 = platform2.step(
            uid=step,
            rankMode=rankModes[1],
            viewMode=viewMode,
            evalMethod="upvote_only",
            numFree=num_free,
            n_showed=n_showed,
            p_pos=p_pos,
            user_c=user_c,
            tau=tau,
            c=c)
        happys1[step % num_consec] = happy1
        happys2[step % num_consec] = happy2
        converge = np.all(
            np.abs(happys1 - happys0[np.arange(num_consec) - 1]) < tol_cnvrg)
        close = np.all(np.abs(happy1 - happy2) < tol_close)
        if converge and close:
            # quality converges and difference < tolerance
            print(tau, c, step, happy1, happy2)
            return tau, c, step, happy1, happy2
        step += 1
    print(tau, c, step, happy1, happy2)
    return tau, c, step, happy1, happy2


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
print("tau", "c", "step", rankModes[0], rankModes[1])

# Simulate
results = []
for i in range(len(tau_and_cs)):
    result = pool.map_async(simulate, zip(items, users, repeat(tau_and_cs[i])))
    results.append(result)

# Output
for result in results:
    happys = result.get()
    tau, c = happys[0][0], happys[0][1]
    with open('outputs/output_tau{}_c{}'.format(tau, c), 'w') as f:
        for happy in happys:
            f.write(', '.join(map(str, happy)) + '\n')
    happys = np.array(happys)
    with open('outputs/output_all_mean', 'a') as f:
        f.write(', '.join(map(str, np.mean(happys, axis=0))) + '\n')
    with open('outputs/output_all_median', 'a') as f:
        f.write(', '.join(map(str, np.median(happys, axis=0))) + '\n')

t_done = time.time()
print("-----Simulation takes {:.4f}s".format(t_done - t_ini))
