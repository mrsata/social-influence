import numpy as np
import random


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
                key=lambda x: self.items[x].getVotes(),
                reverse=True)
        elif mode == 'ucb':
            ranking = sorted(
                self.items.keys(),
                key=
                lambda x: self.items[x].getQuality() /
                np.sqrt(self.items[x].getViews()) + 1000,
                reverse=True)
            # In case of zero division, sort by quality if item.getViews() == 0
        else:
            raise Exception("Unexpected rank mode")
        self.itemRanking = ranking
        self.placeOfItems = [ranking.index(i) for i in self.items.keys()]
        for i,key in enumerate(self.items.keys()):
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

    def run(self, mode, evalMethod='abs_quality', run_mode='all'):
        # Run a simulation with given mode of viewing policies from
        # ['all', 'random', 'position', 'views', 'upvotes']
#        print('viewMode: ' + run_mode)
        if run_mode == 'all':
            # Permutates users with items
            for uid in self.users.keys():
                for iid in self.itemPlacement:
                    evalutaion = self.users[uid].view(self.items[iid],evalMethod)
                    self.viewHistory[iid][uid] += 1
                    self.items[iid].views += 1
                    if evalutaion:
                        self.evalHistory[iid][uid] = evalutaion
                        self.items[iid].setVotes(evalutaion)
                self.rankItems(mode=mode)
                self.placeItems(mode='all')
        else:
            # Permutates users with items
            for uid in self.users.keys():
                for iid in self.itemPlacement:
                    evalutaion = self.users[uid].view(self.items[iid],evalMethod)
                    self.viewHistory[iid][uid] += 1
                    self.items[iid].views += 1
                    if evalutaion:
                        self.evalHistory[iid][uid] = evalutaion
                        self.items[iid].setVotes(evalutaion)
        # TODO: finish all different modes
#        elif mode == 'random':
#            raise Exception("Viewing type " + mode + " not yet implemented")
#        elif mode == 'position':
#            raise Exception("Viewing type " + mode + " not yet implemented")
#        elif mode == 'views':
#            raise Exception("Viewing type " + mode + " not yet implemented")
#        elif mode == 'upvotes':
#            raise Exception("Viewing type " + mode + " not yet implemented")
#        else:
#            raise Exception("Unexpected view mode")

        return self.viewHistory, self.evalHistory
