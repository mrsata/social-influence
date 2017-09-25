import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import xlwt

from item import Item
from user import User
from plat import Platform
from measurements import *

parser = argparse.ArgumentParser()
parser.add_argument('--rankMode', type=str, default='quality')
parser.add_argument('--placeMode', type=str, default='all')
parser.add_argument('--viewMode', type=str, default='all')
args = parser.parse_args()

random.seed(123)
num_item = 10
num_user = 100
items = {}
users = {}
quality_range = range(1, 6)  # quality of 1~5
ability_range = range(1, 6)  # ability of 1~5

# Initialization of items
for i in range(num_item):
    q = random.choice(quality_range)
    items[i] = Item(i, q)

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

#plt.imshow(evalHistory, cmap=plt.cm.Blues, interpolation='nearest')
#plt.title('evalHistory')
#plt.xlabel('user')
#plt.ylabel('item')
#plt.colorbar()
#plt.show()

final_list = [itm for itm in platform.items.values()]
final_ranking = [i+1 for i in platform.itemPlacement]
#print (final_list[0].getQuality())
perfmeas = kendallTauDist(final_list,final_ranking,rank_std="random")
print ("Final ranking: ",perfmeas['final_rank'])
print ("Expected ranking: ",perfmeas['exp_rank'])
print ("Kendall tau distance:",perfmeas['dist'])