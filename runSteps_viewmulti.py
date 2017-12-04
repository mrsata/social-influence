from __future__ import division
from copy import deepcopy
import random
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import time
from item import Item
from user import User
from plat2d import Platform
import os

rdm_quality = False  # assign item quality randomly
#plotQuality = False  # plot item quality
num_free = 1  # number of free views upon initialization
num_runs = 10  # number of realizations
num_item = 50  # total number of items
num_user = 500000  # total number of users (time)
lower, upper = 0, 1  # lower and upper bound of item quality
mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
ability_range = range(1, 6)  # ability of 1~5
#rankModes = ['quality', 'ucb']
viewModes = ['first', 'position','multi']
viewMode = viewModes[2]  # how platform displays items to users
p_pos = 0.5  # ratio of positional preference in user's choice
# p_pos=1 has only positional prefernece

## PARAMETERS
parameters = ['tau','uc','nshow','lc']
# fixed values
fixed_tau, fixed_uc, fixed_nshow, fixed_lc = 1, 1, 1, 1 
# PARAMETERS 
#paras = np.linspace(0.02,1,50)
#paras = np.linspace(0.02,0.3,15)
#paras = np.linspace(0.32,0.6,15)
#paras = np.linspace(0.62,0.9,15)
paras = np.linspace(0.12,0.2,5)
#paras = np.linspace(0.2,1,10)

# convergence condition
tol_cnvrg = 1e-5
#tol_close = 0.0001
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
def simulate(its, urs, para):
    # np.random.seed(123)
    items = its
    users = urs
    tau, c, n_showed, user_c = fixed_tau, fixed_uc, fixed_nshow, fixed_lc
    n_showed = para
    platform1 = Platform(items=deepcopy(items), users=users)
#    platform2 = Platform(items=deepcopy(items), users=users)
    happys1 = np.zeros(num_consec)
    hs1 = np.zeros(num_consec)
    step = 0
    h1 = 0
    while step < num_user:
        happys0 = happys1.copy()
        hs0 = hs1.copy()
        happy1,h1 = platform1.step(
            uid=step,
            prevHappy = h1,
            rankMode='ucb',
            viewMode=viewMode,
            evalMethod="upvote_only",
            numFree=num_free,
            n_showed=n_showed,
            p_pos=p_pos,
            user_c=user_c,
            tau=tau,
            c=c)
#        if step<50:
#            print ("Step",step,"|",h1)
        happys1[step % num_consec] = happy1
        hs1[step % num_consec] = h1
        converge = np.all(
            np.abs(happys1 - happys0[np.arange(num_consec) - 1]) < tol_cnvrg)
        converge = np.all(
            np.abs(hs1 - hs0[np.arange(num_consec) - 1]) < tol_cnvrg)
        if converge and h1!=0:
            # quality converges and difference < tolerance
            totalViews = np.sum(platform1.items[1])
            totalUpvotes = np.sum(platform1.items[2])
            ratio = totalUpvotes/totalViews
            print(para, step, h1, totalViews, ratio)
            return para, step, h1, totalViews, ratio
        step += 1
        
        totalViews = np.sum(platform1.items[1])
        totalUpvotes = np.sum(platform1.items[2])
        ratio = totalUpvotes/totalViews
    print(para, step, h1, totalViews, ratio)
    return para, step, h1, totalViews, ratio


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
zipped = zip(items, users)
items, users = zip(*zipped)
for i in range(len(seeds)):
    qualities = [itm.getQuality() for itm in items[i].values()]
    mean_quality = np.mean(qualities)

t_ini = time.time()
print("-----Initialization takes %.4fs" % (t_ini - t0))
print("show", "step", 'happy(quality)', '#Views', '#Upvotes/#Views')


# Simulate
results = []
#m,n = paras_grid.shape
for i_run in range(num_runs):
    itms = items[i_run]
    usrs = users[i_run]
    result_realization = []
    for i in range(len(paras)):
        result = simulate(itms, usrs, paras[i])
        result_realization.append(result)
    results.append(result_realization)
    
### Output
results = np.array(results)
outputpath = 'output'
if not os.path.isdir(outputpath):
   os.makedirs(outputpath)
savepath = outputpath+'/show'
if not os.path.isdir(savepath):
   os.makedirs(savepath)
for i in range(len(paras)):
    para = paras[i]
    with open((savepath+'/show{0}.txt').format(para), 'w') as f:
        f.write(str(para)+"  number of realizations "+str(num_runs)+"\n")
        f.write("   #step    happiness(quality)    #views    #upvotes/#views \n")
        f.write('\n'.join(map(str, results[:,i,1:])) + '\n')
        f.write("mean  "+str(np.mean(results[:,i,1:], axis=0))+"\n")
        f.write("median"+str(np.median(results[:,i,1:], axis=0))+"\n")
        f.write("max   "+str(np.max(results[:,i,1:], axis=0))+"\n")
        f.write("min   "+str(np.min(results[:,i,1:], axis=0))+"\n")

#t_done = time.time()
#print("-----Simulation takes %.4fs" % (t_done - t_ini))
