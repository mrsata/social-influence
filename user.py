import random


class User(object):
    """Abstract base class for users

    A user is an agent that can decide whether to view and to upvote the item.

    Property:
        uid
        ability
    Function:
        view
        eval

    """

    def __init__(self, uid, ability):
        self.uid = uid
        self.ability = ability

    def view(self, item):
        # Users evaluate an item as long as they view it
        # TODO: change the user's intent to evaluate & create
        evalutation = self.evaluate(item) if random.random() > 0 else 0
        return evalutation

    def evaluate(self, item):
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
        return item.getQuality() - 2.5

    def create(self):
        pass

    def getID(self):
        return self.uid

    def getAbility(self):
        return self.ability
