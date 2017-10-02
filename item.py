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
        self.votes = [0, 0]
        self.place = None

    def getID(self):
        return self.iid

    def getQuality(self):
        return self.quality

    def getViews(self):
        return self.views

    def getVotes(self):
        return self.votes
    
    def getUpVotes(self):
        return self.votes[0]
    
    def getDownVotes(self):
        return self.votes[1]

    def getPlace(self):
        return self.place

    def setVotes(self, evaluation):
        if evaluation == 0:
            raise Exception("Setting vote for unevaluated item with id: " +
                            str(self.iid))
        elif evaluation > 0:
            self.votes[0] += 1
        else:
            self.votes[1] += 1

    def setPlace(self, p):
        self.place = p
