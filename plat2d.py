import numpy as np
import random
from measurements import *


class Platform(object):
    """Abstract base class for the platform in 2d dimension

    A platform is an environment where users interacts with items.

    Property:
        int num_item
        int num_user
        np.ndarray items
        np.ndarray viewHistory
        np.ndarray evalHistory
        np.ndarray itemRanking
        np.ndarray itemPlacement
    Function:
        rankItems()
        placeItems()
        run()

    """

    def __init__(self, items, users):
        self.num_item = len(items)
        self.num_user = len(users)
        self.items = np.zeros((5, self.num_item))
        for k, v in items.items():
            self.items[0, k] = v.getQuality()
            self.items[1, k] = v.getViews()
            self.items[2, k] = v.getUpVotes()
            self.items[3, k] = v.getDownVotes()
        # rows: quality, views, upvotes, downvotes, ranking
        self.users = users
        self.viewHistory = np.zeros((self.num_item, self.num_user))
        self.evalHistory = np.zeros((self.num_item, self.num_user))
        self.itemRanking = np.zeros(self.num_item)
        self.itemPlacement = np.zeros(self.num_item)
        self.perfmeas = []
        self.viewProb = .97**(np.arange(self.num_item) + 1)
        self.viewProb = self.viewProb / np.sum(self.viewProb)

    def rankItems(self, mode='random', t=0):
        # Rank items with given mode of ranking policies from
        # ['random', 'quality', 'ucb', 'lcb', 'upvotes', 'views']
        if mode == 'random':
            ranking = np.arange(self.num_item)
            np.random.shuffle(ranking)
        elif mode == 'quality':
            ranking = np.argsort(-self.items[0])
        elif mode == 'views':
            ranking = np.argsort(-self.items[1])
        elif mode == 'upvotes':
            ratio = np.true_divide(self.items[2], self.items[1])
            ranking = np.argsort(-ratio)
        elif mode == 'lcb':
            lower = confidenceBound(self.items, self.num_user)[0]
            ranking = np.argsort(-lower)
        elif mode == 'ucb':
            upper = confidenceBound(self.items, self.num_user)[1]
            ranking = np.argsort(-upper)
        else:
            raise Exception("Unexpected rank mode")
        self.itemRanking = ranking
        self.items[-1] = np.argsort(ranking)
        return ranking

    def placeItems(self, mode='all'):
        if mode == 'all':
            placement = self.itemRanking
        else:
            raise Exception("Unexpected place mode")
        self.itemPlacement = placement
        return placement

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
        # Ranking in descending quality order
        desq_rank = np.argsort(-self.items[0])
        expected_top_K = desq_rank[0:perfmeasK]
        # Run Start
        for uid in range(self.num_user):
            if viewMode == 'first':
                # user view the first in ranking
                iid = self.itemRanking[0]
            else:  # viewMode == 'position'
                # OLD: user view a single item with position bias
                itm_place = np.random.choice(self.num_item, 1, p=self.viewProb)
                iid = self.itemRanking[itm_place[0]]

            self.viewHistory[iid][uid] += 1
            self.items[1, iid] += 1

            if evalMethod == "abs_quality":
                evaluation = self.items[0, iid]
            elif evalMethod == "rel_quality":
                evaluation = self.items[0, iid] + np.random.normal(0, 0.1)
            else:  # "upvote_only":
                evaluation = 1 if np.random.rand() < self.items[0, iid] else -1
            if evaluation:
                self.evalHistory[iid][uid] = evaluation
                if evaluation > 0:
                    self.items[2, iid] += 1
                else:
                    self.items[3, iid] += 1

            ########
            # first 500 runs random to get initial data
            if uid < 0:
                self.rankItems(mode='random')
            else:
                self.rankItems(mode=rankMode, t=uid + 1)
            ########
            self.placeItems(mode='all')

            time = uid + 1
            # Happiness
            sum_upvotes = np.sum(self.items[2])
            upvotes_t = sum_upvotes / (time + 1 * self.num_item)
            happy = upvotes_t
            # Kendall Tau Distance
            if rankMode == "random":
                # Reorder the final list in #votes order if ranking strategy is random -> Ranking in descending upvotes order
                ratio = np.true_divide(self.items[2], self.items[1])
                final_rank = np.argsort(-ratio)
            else:
                final_rank = self.itemRanking
            # Calculate the Kendall tau distance
            tau, p_value = stats.kendalltau(desq_rank, final_rank)
            # Top K Percentage
            actual_top_K = final_rank[0:perfmeasK]
            in_top_K = set(expected_top_K).intersection(actual_top_K)
            topK = len(in_top_K) / perfmeasK
            perfmea = {'ktd': tau, 'topK': topK, 'happy': happy}
            self.perfmeas.append(perfmea)

        return self.perfmeas


def confidenceBound(items, T):
    c = 1
    ratio = np.true_divide(items[2], items[1])
    bound = np.true_divide(np.log(T), items[1])
    bound = c * np.sqrt(bound)
    lower = ratio - bound
    upper = ratio + bound
    return (lower, upper)
