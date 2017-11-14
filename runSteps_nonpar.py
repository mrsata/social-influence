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
n_showed = 0.1  # portion of items displayed by the platform
p_pos = 1  # ratio of positional preference in user's choice
# p_pos=1 has only positional prefernece
user_c = 0.5  # coeff of user's lcb
parameters = ['tau','c','n_showed']
paras = [parameters[1],parameters[2]]
para_ranges = []
for i in range(2):
    if paras[i]=='tau':
        para_ranges.append(np.linspace(0,2,3))
    elif paras[i]=='c':
        para_ranges.append(np.linspace(0,2,9))
    elif paras[i]=='n_showed':
        para_ranges.append(np.linspace(0.1,1,10))
mesh1,mesh2 = np.meshgrid(para_ranges[0], para_ranges[1])
paras_grid = np.dstack(
    (mesh1,mesh2)).reshape(-1, 2)
tol_cnvrg = 0.005
tol_close = 0.005
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
def simulate(its, urs, paras, para_i):
    # np.random.seed(123)
    items = its
    users = urs
    if ('tau' in paras) and ('c' in paras):
        tau, c = para_i
        n_showed = 1
    elif ('c' in paras) and ('n_showed' in paras):
        c, n_showed = para_i
        tau = 1
    else:
        tau, c, n_showed = 1, 1, 1
    platform1 = Platform(items=deepcopy(items), users=users)
    platform2 = Platform(items=deepcopy(items), users=users)
    happys1 = np.zeros(num_consec)
    happys2 = np.zeros(num_consec)
    step = 0
    while step < num_user:
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
            print(para_i[0], para_i[1], step, happy1, happy2)
            return para_i[0], para_i[1], step, happy1, happy2
        step += 1
    print(para_i[0], para_i[1], step, happy1, happy2)
    return para_i[0], para_i[1], step, happy1, happy2


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
print(paras[0], paras[1], "step", rankModes[0], rankModes[1])


# Simulate
results = []
m,n = paras_grid.shape
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
for i in range(m):
    p1,p2 = paras_grid[i]
    with open(('outputs/output_'+paras[0]+'{0}_'+paras[1]+'{1}.txt').format(p1, p2), 'w') as f:
        f.write(paras[0]+" "+str(p1)+"  "+paras[1]+" "+str(p2)+"  number of realizations "+str(num_runs)+"\n")
        f.write("   #step    happiness(quality)    happiness(ucb) \n")
        f.write('\n'.join(map(str, results[:,i,2:])) + '\n')
        f.write("mean  "+str(np.mean(results[:,i,2:], axis=0))+"\n")
        f.write("median"+str(np.median(results[:,i,2:], axis=0))+"\n")
        f.write("max   "+str(np.max(results[:,i,2:], axis=0))+"\n")
        f.write("min   "+str(np.min(results[:,i,2:], axis=0))+"\n")

#t_done = time.time()
#print("-----Simulation takes %.4fs" % (t_done - t_ini))
