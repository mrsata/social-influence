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
print ("-----Start")
parser = argparse.ArgumentParser()
parser.add_argument('--rankMode', type=str, default='upvotes')
parser.add_argument('--placeMode', type=str, default='all')
parser.add_argument('--viewMode', type=str, default='all')
args = parser.parse_args()
rdm_quality = False  # assign item quality randomly
plotPerf = True  # plot performance
plotQuality = False  # plot item quality
plotHistory = False  # plot rating history

random.seed(123)
fig_idx = 0
num_item = 50
num_user = 10000
items = {}
users = {}
lower, upper = 0, 1  # lower and upper bound of item quality
ability_range = range(1, 6)  # ability of 1~5
K = 20 # number of items for performance measurement "top K in expected top K"
rankMode = args.rankMode

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
# assign qualities to items
for i in range(num_item):
    if rdm_quality:
        q = random.uniform(lower, upper)
    else:
        q = qualities[i]
    items[i] = Item(i, q)
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
print ("-----Initialization takes",t_ini-t0)

#********** Run simulation once
platform = Platform(items=items, users=users)
platform.rankItems(mode=args.rankMode)
platform.placeItems(mode=args.placeMode)
viewHistory, evalHistory = platform.run(
    mode=rankMode, evalMethod="upvote_only", run_mode='all',perfmeasK = K)
accum_evals = np.cumsum(evalHistory, axis=1)  # accumulation of evaluations

#********** Performance Measurements
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
happy = happiness(final_list,count="upvotes")
#happiness = totalNumUpVotes/num_user
print ()
print("User happiness:", happy)

# Timing
print ()
t_done = time.time()
print ("-----Simulation takes",t_done-t_ini)

#***** Trends of performances
perfmeas1 = platform.perfmeas
ktds1 = [pf['ktd'] for pf in perfmeas1]
topKs1 = [pf['topK'] for pf in perfmeas1]
happy1 = [pf['happy'] for pf in perfmeas1]

#********** Plotting
if plotPerf:
    # kendall tau distance
    fig_idx += 1
    plt.figure(fig_idx)
    plt.plot(ktds1,label='rank by %s'%(rankMode))
    plt.title('kendall tau distance VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('kendall tau distance')
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()

    # top k in k
    fig_idx += 1
    plt.figure(fig_idx)
    plt.plot(topKs1,label='rank by %s'%(rankMode))
    plt.title('percentage of top %d in %d VS. time'%(K,K))
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('percentage of top %d in %d '%(K,K))
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()
    
    # user happiness
    fig_idx += 1
    plt.figure(fig_idx)
    plt.plot(happy1,label='rank by %s'%(rankMode))
    plt.title('user happiness VS. time')
    plt.minorticks_on()
    plt.xlabel('time')
    plt.ylabel('user happiness')
    plt.ylim([0,1.1])
    plt.legend()
    plt.grid()
    plt.show()


if plotHistory:
    # Plot the evalution history
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
