import numpy as np
import scipy.stats as st
import random
from measurements import *


class Platform(object):
    """Abstract base class for the platform

    A platform is an environment where users interacts with items.

    Property:
        dict items
        dict users
        int num_item
        int num_user
        np.ndarray viewHistory
        np.ndarray evalHistory
        list itemRanking
        list itemPlacement
    Function:
        rankItems()
        placeItems()
        run()

    """

    def __init__(self, items, users):
        self.items = items
        self.users = users
        self.num_item = len(items)
        self.num_user = len(users)
        self.viewHistory = np.zeros((self.num_item, self.num_user))
        self.evalHistory = np.zeros((self.num_item, self.num_user))
        self.itemRanking = None
        self.itemPlacement = None
        self.placeOfItems = None
        self.perfmeas = list()

    def rankItems(self, mode='quality'):
        # Rank items with given mode of ranking policies from
        # ['random', 'quality', 'views', 'upvotes', 'ucb']
        #        print('rankMode: ' + mode)
        if mode == 'random':
            ranking = list(self.items.keys())
            random.shuffle(ranking)
        elif mode == 'quality':
            ranking = sorted(
                self.items.keys(),
                key=lambda x: self.items[x].getQuality(),
                reverse=True)
        elif mode == 'views':
            ranking = sorted(
                self.items.keys(),
                key=lambda x: self.items[x].getViews(),
                reverse=True)
        elif mode == 'upvotes':
            ranking = sorted(
                self.items.keys(),
                key=lambda x: self.items[x].getUpVotes()/sum(self.items[x].getVotes()) if self.items[x].getUpVotes() else 0,
                reverse=True)
        elif mode == 'lcb':
            ranking = sorted(
                self.items.keys(),
                key=
                lambda x: wilsonScoreInterval(self.items[x].getUpVotes(), self.items[x].getDownVotes())[0],
                reverse=True)
        elif mode == 'ucb':
            ranking = sorted(
                self.items.keys(),
                key=
                lambda x: wilsonScoreInterval(self.items[x].getUpVotes(), self.items[x].getDownVotes())[1],
                reverse=True)
        else:
            raise Exception("Unexpected rank mode")
        self.itemRanking = ranking
        self.placeOfItems = [ranking.index(i) for i in self.items.keys()]
        for i, key in enumerate(self.items.keys()):
            self.items[key].setPlace(self.placeOfItems[i])
        return ranking

    def placeItems(self, mode='all'):
        #        print('placeMode: ' + mode)
        if mode == 'all':
            placement = self.itemRanking
        else:
            raise Exception("Unexpected place mode")
        self.itemPlacement = placement
        return placement

    def run(self, mode, evalMethod='abs_quality', run_mode='all',
            perfmeasK=10):
        # Run a simulation with given mode of viewing policies from
        # ['all', 'random', 'position', 'views', 'upvotes']
        #        print('viewMode: ' + run_mode)
        if run_mode == 'all':
            # Permutates users with items
            for uid in self.users.keys():
                # user view the first in ranking
                # if self.itemRanking:
                #    iid = self.itemRanking[0]
                # else:
                #    iid = sorted(
                # self.items.keys(),
                # key=lambda x: self.items[x].getQuality(),
                # reverse=True)[0]
                # OLD: user view a single item at a time
                viewProb = [0.97**(i + 1) for i in range(0, self.num_item)]
                viewProb = viewProb / np.sum(viewProb)
                itm_place = np.random.choice(self.num_item, 1, p=viewProb)
                if self.itemRanking:
                    iid = self.itemRanking[itm_place[0]]
                else:
                    iid = np.random.choice(self.num_item, 1)

                self.viewHistory[iid][uid] += 1
                self.items[iid].views += 1
                evalutaion = self.users[uid].view(self.items[iid], evalMethod)
                if evalutaion:
                    self.evalHistory[iid][uid] = evalutaion
                    self.items[iid].setVotes(evalutaion)
                # OLD: user view multiple items at a time
                # for iid in self.itemPlacement:
                #    evalutaion = self.users[uid].view(self.items[iid],
                #                                      evalMethod)
                #    self.viewHistory[iid][uid] += 1
                #    self.items[iid].views += 1
                #    if evalutaion:
                #        self.evalHistory[iid][uid] = evalutaion
                #        self.items[iid].setVotes(evalutaion)

                ########
                # first 500 runs random to get initial data
                if uid < 2000:
                    self.rankItems(mode='random')
                else:
                    self.rankItems(mode=mode)
                ########

                self.placeItems(mode='all')
                #***** measure the performances after first 10 runs

                cur_list = [itm for itm in self.items.values()]
                # kendall Tau Distance
                ktd = kendallTauDist(
                    cur_list,
                    final_placements=[i + 1 for i in self.placeOfItems],
                    rank_std="random")
                # Top K Percentage
                topK = topKinK(
                    cur_list,
                    K=perfmeasK,
                    final_order=self.itemRanking,
                    rank_std="random")
                # User Happiness
                happy = happiness(cur_list, uid + 1, count="upvotes")

                perfmea = {
                    'ktd': ktd['dist'],
                    'topK': topK['percent'],
                    'happy': happy
                }
                self.perfmeas.append(perfmea)
                #**********
        else:
            # Permutates users with items
            for uid in self.users.keys():
                for iid in self.itemPlacement:
                    evalutaion = self.users[uid].view(self.items[iid],
                                                      evalMethod)
                    self.viewHistory[iid][uid] += 1
                    self.items[iid].views += 1
                    if evalutaion:
                        self.evalHistory[iid][uid] = evalutaion
                        self.items[iid].setVotes(evalutaion)

        return self.viewHistory, self.evalHistory


def wilsonScoreInterval(ups, downs, confidence=.9):
    n = ups + downs
    if n == 0:
        return (0, 0)
    # z = st.norm.ppf((1 + confidence) / 2)
    # z = 1.2815515655446004  # confidence = .8
    z = 1.6448536269514722  # confidence = .9
    phat = ups / n
    lower = ((phat + z * z / (2 * n) - z * np.sqrt(
        (phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
    upper = ((phat + z * z / (2 * n) + z * np.sqrt(
        (phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
    return (lower, upper)
