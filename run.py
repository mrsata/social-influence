import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
#import xlwt

from item import Item
from user import User
from plat import Platform
from measurements import *

parser = argparse.ArgumentParser()
parser.add_argument('--rankMode', type=str, default='quality')
parser.add_argument('--placeMode', type=str, default='all')
parser.add_argument('--viewMode', type=str, default='all')
args = parser.parse_args()
rdm_quality = False
plot = True

random.seed(123)
fig_idx = 0
num_item = 20
num_user = 50
items = {}
users = {}
lower, upper = 0, 1  # lower and upper bound of item quality
ability_range = range(1, 6)  # ability of 1~5

# Initialization of items
if not rdm_quality:  # assume item qualities follow a normal distribution between 0~1
    mu, sigma = 0.5, 0.3  # mean and standard deviation of item quality
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    qualities = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size=num_item)
    if plot:
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

# Initialization of users
for i in range(num_user):
    a = random.choice(ability_range)
    users[i] = User(i, a)

# Run simulation once
print("num_item: {}, num_user: {}".format(num_item, num_user))
platform = Platform(items=items, users=users)
platform.rankItems(mode=args.rankMode)
platform.placeItems(mode=args.placeMode)
# Currently available mode: 'random', 'quality', 'views', 'upvotes', 'ucb'
# Currently available evalMethod: 'abs_quality', 'rel_quality', 'upvote_only'
viewHistory, evalHistory = platform.run(
    mode='quality', evalMethod="upvote_only", run_mode='all')
accum_evals = np.cumsum(evalHistory, axis=1)  # accumulation of evaluations

# Measure the performance
final_list = [itm for itm in platform.items.values()]  # list of items
final_ranking = [i + 1 for i in platform.placeOfItems
                 ]  # places of items in final display
perfmeas = kendallTauDist(final_list, final_ranking, rank_std="upvotes")
print("Final ranking: ", perfmeas['final_rank'])
print("Expected ranking: ", perfmeas['exp_rank'])
print("Kendall tau distance:", perfmeas['dist'])

if plot:
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

    # Plot the accumulated evalution history
    fig_idx += 1
    plt.figure(fig_idx)
    plt.imshow(accum_evals, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title('Accumulated Evaluation History')
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
