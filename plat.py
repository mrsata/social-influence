import numpy as np
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

    def rankItems(self, mode='random', t=0):
        # Rank items with given mode of ranking policies from
        # ['random', 'quality', 'ucb', 'lcb', 'upvotes', 'views']
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
                key=lambda x: self.items[x].getUpVotes() / self.items[x].getViews() if self.items[x].getViews() else 0,
                reverse=True)
        elif mode == 'lcb':
            ranking = sorted(
                self.items.keys(),
                key=
                lambda x: confidenceBound(self.items[x].getUpVotes(), self.items[x].getViews(), t, self.num_user)[0],
                reverse=True)
        elif mode == 'ucb':
            ranking = sorted(
                self.items.keys(),
                key=
                lambda x: confidenceBound(self.items[x].getUpVotes(), self.items[x].getViews(), t, self.num_user)[1],
                reverse=True)
        else:
            raise Exception("Unexpected rank mode")
        self.itemRanking = ranking
        self.placeOfItems = [ranking.index(i) for i in self.items.keys()]
        for i, key in enumerate(self.items.keys()):
            self.items[key].setPlace(self.placeOfItems[i])
        return ranking

    def placeItems(self, mode='all'):
        if mode == 'all':
            placement = self.itemRanking
        else:
            raise Exception("Unexpected place mode")
        self.itemPlacement = placement
        return placement

    # TODO: may seperate warmup from real run
    # @staticmethod
    # def warmup(items, num_iter=100, mode='random', evalMethod='abs_quality'):
    #     user = User(0, 0)
    #     itemRanking = list(items.keys())
    #     for i in range(num_iter):
    #         random.shuffle(itemRanking)
    #         iid = itemRanking[0]
    #         items[iid].views += 1
    #         evalutaion = user.evaluate(items[iid], evalMethod)
    #         if evalutaion:
    #             items[iid].setVotes(evalutaion)

    def run(self,
            rankMode='random',
            viewMode='first',
            evalMethod='upvote_only',
            perfmeasK=10):
        # Run a simulation with given mode of viewing policies from
        # ['first', 'position']
        # Initialization
        self.rankItems(mode=rankMode)
        self.placeItems(mode='all')
        # Run Start
        for uid in self.users.keys():
            if viewMode == 'first':
                # user view the first in ranking
                if self.itemRanking:
                    iid = self.itemRanking[0]
                else:
                    iid = sorted(
                        self.items.keys(),
                        key=lambda x: self.items[x].getQuality(),
                        reverse=True)[0]
            else:  # viewMode == 'position'
                # OLD: user view a single item with position bias
                viewProb = [0.97**(i + 1) for i in range(0, self.num_item)]
                viewProb = viewProb / np.sum(viewProb)
                itm_place = np.random.choice(self.num_item, 1, p=viewProb)
                if self.itemRanking:
                    iid = self.itemRanking[itm_place[0]]
                else:
                    iid = np.random.choice(self.num_item, 1)

            self.viewHistory[iid][uid] += 1
            self.items[iid].views += 1
            evalutaion = self.users[uid].evaluate(self.items[iid], evalMethod)
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
            if uid < 0:
                self.rankItems(mode='random')
            else:
                self.rankItems(mode=rankMode, t=uid + 1)
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

        return self.perfmeas


# Old: inappropriate ucb with wsi
# def wilsonScoreInterval(ups, downs, confidence=.9):
#     n = ups + downs
#     if n == 0:
#         return (0, 999)
#     # z = st.norm.ppf((1 + confidence) / 2)
#     # z = 1.2815515655446004  # confidence = .8
#     z = 1.6448536269514722  # confidence = .9
#     phat = ups / n
#     lower = ((phat + z * z / (2 * n) - z * np.sqrt(
#         (phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
#     upper = ((phat + z * z / (2 * n) + z * np.sqrt(
#         (phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
#     return (lower, upper)


def confidenceBound(ups, views, t, T):
    if views == 0:
        return (9999, 9999)
    c = 1
    p = ups / views
    lower = p - c * np.sqrt(np.log(T) / views)
    upper = p + c * np.sqrt(np.log(T) / views)
    return (lower, upper)
