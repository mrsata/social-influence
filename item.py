class Item(object):
    """Abstract base class for items

    An item is something to be presented for users.

    Property:
        iid
        quality
        views
        votes
    Function:
        setVotes()

    """

    def __init__(self, iid, quality):
        self.iid = iid
        self.quality = quality
        self.views = 0
        self.votes = 0

    def getID(self):
        return self.iid

    def getQuality(self):
        return self.quality

    def getViews(self):
        return self.views

    def getVotes(self):
        return self.votes

    def setVotes(self, evaluation):
        if evaluation == 0:
            raise Exception("Setting vote for unevaluated item with id: " +
                            str(self.iid))
        elif evaluation > 0:
            self.votes += 1
        else:
            self.votes -= 1
