import random
import numpy as np


class User(object):
    """Abstract base class for users

    A user is an agent that can decide whether to view and to upvote the item.

    Property:
        uid
        ability
    Function:
        view()
        evaluate()
        create()

    """

    def __init__(self, uid, ability):
        self.uid = uid
        self.ability = ability

    def view(self, item, evalMethod):
        # Users evaluate an item as long as they view it
        # TODO: change the user's intent to evaluate & create
        # item_place = item.getPlace()
        # # The higher the place, the more likely the user chooses to view it.
        # prob = 0.97**(item_place+1) # 0.97^place
        # evaluation = self.evaluate(item,evalMethod) if random.random() < prob else 0
        evaluation = random.random() < item.getQuality()*1.5 # prob of evaluate single item
        return evaluation

    def evaluate(self, item, method='abs_quality'):
        # Evaluate the item naively using the item's true quality
        # TODO: change the evaluation policy
        ''' Random Policy:
        rdm1 = random.random()
        rdm2 = random.random()
        qlt = item.getQuality()
        a = qlt / 5 * 0.8
        if rdm1 < a or rdm2 < 0.2:
            # upvote
        elif rdm2 > 0.8:
            # downvote
        else:
            # do nothing
        '''
        item_q = item.getQuality()
        if method=="abs_quality":
            evaluation = item_q
        elif method=="rel_quality":
            evaluation = item_q + np.random.normal(0, 0.1)
        else: # "upvote_only":
            evaluation = 1 if random.random()<item_q else -1
            # evaluation = 1 if (0.9*item_q + np.random.normal(0, 0.2))>0.7 else 0
        return evaluation

    def create(self):
        pass

    def getID(self):
        return self.uid

    def getAbility(self):
        return self.ability
