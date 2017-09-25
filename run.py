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

random.seed(123)
num_item = 20
num_user = 30
items = {}
users = {}
lower, upper = 0, 1 # lower and upper bound of item quality
ability_range = range(1, 6)  # ability of 1~5

# Initialization of items
if not rdm_quality: # assume item qualities follow a normal distribution between 0 and 1
    mu, sigma = 0.5, 0.3 # mean and standard deviation of item quality
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    qualities = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size=num_item)
    fig, ax = plt.subplots(1, 1)
    ax.hist(qualities,normed=True)
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
print("Qualities: \n mean {} \n min  {} \n max  {}".format(mean_quality, min_quality, max_quality))

# Initialization of users
for i in range(num_user):
    a = random.choice(ability_range)
    users[i] = User(i, a)

# Run simulation once
print("num_item: {}, num_user: {}".format(num_item, num_user))
platform = Platform(items=items, users=users)
platform.rankItems(mode=args.rankMode)
platform.placeItems(mode=args.placeMode)
viewHistory, evalHistory = platform.run(mode=args.viewMode)

# Plot the viewing and evalution history
plt.imshow(evalHistory, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('evalHistory')
plt.xlabel('user')
plt.ylabel('item')
plt.colorbar()
plt.show()

# Measure the performance
final_list = [itm for itm in platform.items.values()]
final_ranking = [i+1 for i in platform.itemPlacement]
#print (final_list[0].getQuality())
perfmeas = kendallTauDist(final_list,final_ranking,rank_std="quality")
print ("Final ranking: ",perfmeas['final_rank'])
print ("Expected ranking: ",perfmeas['exp_rank'])
print ("Kendall tau distance:",perfmeas['dist'])