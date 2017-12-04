from __future__ import division
from copy import deepcopy
import random
import numpy as np
import scipy.stats as stats
import argparse
import time
import os
from item import Item
from user import User
from plat2d import Platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tau', nargs='*', type=float)
parser.add_argument('-u', '--uc', nargs='*', type=float)
parser.add_argument('-n', '--nshow', nargs='*', type=float)
parser.add_argument('-l', '--lc', nargs='*', type=float)
args = parser.parse_args()

rdm_quality = False  # assign item quality randomly
num_free = 1  # number of free views upon initialization
num_runs = 1  # number of realizations
num_item = 50  # total number of items
num_user = 500000  # total number of users (time)
lower, upper = 0, 1  # lower and upper bound of item quality
mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
ability_range = range(1, 6)  # ability of 1~5
p_pos = 0.5  # ratio of positional preference in user's choice

## PARAMETERS
parameters = ['tau', 'uc', 'nshow', 'lc']
fixed_tau, fixed_uc, fixed_nshow, fixed_lc = 1, 1, 1, 1
paras = [args.tau, args.uc, args.nshow, args.lc]
paras = [parameters[i] for i, para in enumerate(paras) if para != None]
para_ranges = []
for i in range(2):
    if paras[i] == 'tau':
        taus = args.tau if len(args.tau) == 3 else args.tau + [.25]
        para_ranges.append(np.arange(*tuple(taus)))
    elif paras[i] == 'uc':
        ucs = args.uc if len(args.uc) == 3 else args.uc + [.25]
        para_ranges.append(np.arange(*tuple(ucs)))
    elif paras[i] == 'nshow':
        nshows = args.nshow if len(args.nshow) == 3 else args.nshow + [.05]
        para_ranges.append(np.arange(*tuple(nshows)))
    elif paras[i] == 'lc':
        lcs = args.lc if len(args.lc) == 3 else args.lc + [.25]
        para_ranges.append(np.arange(*tuple(lcs)))

mesh1, mesh2 = np.meshgrid(para_ranges[0], para_ranges[1])
paras_grid = np.dstack((mesh1, mesh2)).reshape(-1, 2)

# convergence condition
tol_cnvrg = 1e-5
num_consec = 1000


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
def simulate(its, urs, paras, para_i):
    # np.random.seed(123)
    items = its
    users = urs
    tau, c, n_showed, user_c = fixed_tau, fixed_uc, fixed_nshow, fixed_lc
    if paras == ['tau', 'uc']:
        tau, c = para_i
    elif paras == ['uc', 'nshow']:
        c, n_showed = para_i
    elif paras == ['nshow', 'lc']:
        n_showed, user_c = para_i
    elif paras == ['tau', 'nshow']:
        tau, n_showed = para_i
    elif paras == ['tau', 'lc']:
        tau, user_c = para_i
    elif paras == ['uc', 'lc']:
        c, user_c = para_i
    else:
        tau, c, n_showed, user_c = fixed_tau, fixed_uc, fixed_nshow, fixed_lc
    platform = Platform(items=deepcopy(items), users=users)
    happys1 = np.zeros(num_consec)
    step = 0
    while step < num_user:
        happys0 = happys1
        happy = platform.step(
            uid=step,
            rankMode='ucb',
            viewMode='position',
            evalMethod="upvote_only",
            numFree=num_free,
            n_showed=n_showed,
            p_pos=p_pos,
            user_c=user_c,
            tau=tau,
            c=c)
        happys1[step % num_consec] = happy
        converge = np.all(
            np.abs(happys1 - happys0[np.arange(num_consec) - 1]) < tol_cnvrg)
        if converge:
            # quality converges and difference < tolerance
            print(para_i[0], para_i[1], step, happy)
            return para_i[0], para_i[1], step, happy
        step += 1
    print(para_i[0], para_i[1], step, happy)
    return para_i[0], para_i[1], step, happy


#********** Start
t0 = time.time()
print("-----Start\nnum_runs: {0}\nnum_item: {1}\nnum_user: {2}".format(
    num_runs, num_item, num_user))

# Initialize
seeds = range(num_runs)
items = []
users = []
for i in range(len(seeds)):
    itms, usrs = initialize(seeds[i])
    items.append(itms)
    users.append(usrs)

t_ini = time.time()
print("-----Initialization takes %.4fs" % (t_ini - t0))
print(paras[0], paras[1], "step", 'ucb')

# Simulate
results = []
m, n = paras_grid.shape
for i_run in range(num_runs):
    itms = items[i_run]
    usrs = users[i_run]
    result_realization = []
    for i in range(m):
        result = simulate(itms, usrs, paras, paras_grid[i])
        result_realization.append(result)
    results.append(result_realization)

### Output
results = np.array(results)
outputpath = 'output'
if not os.path.isdir(outputpath):
    os.makedirs(outputpath)
savepath = outputpath + '/' + paras[0] + '&' + paras[1]
if not os.path.isdir(savepath):
    os.makedirs(savepath)
for i in range(m):
    p1, p2 = paras_grid[i]
    with open(
        (savepath + '/' + paras[0] + '{:.2f}_' + paras[1] + '{:.2f}.txt').format(
            p1, p2), 'w') as f:
        f.write(paras[0] + " " + str(p1) + "  " + paras[1] + " " + str(p2) +
                "  number of realizations " + str(num_runs) + "\n")
        f.write("   #step    happiness(quality)    happiness(ucb) \n")
        f.write('\n'.join(map(str, results[:, i, 2:])) + '\n')
        f.write("mean  " + str(np.mean(results[:, i, 2:], axis=0)) + "\n")
        f.write("median" + str(np.median(results[:, i, 2:], axis=0)) + "\n")
        f.write("max   " + str(np.max(results[:, i, 2:], axis=0)) + "\n")
        f.write("min   " + str(np.min(results[:, i, 2:], axis=0)) + "\n")

t_done = time.time()
print("-----Simulation takes %.4fs" % (t_done - t_ini))
