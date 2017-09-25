
class Item(object):
    """Abstract base class for items

    An item is something to be presented for users.

    Property:
        iid
        quality
    To add:
        placement

    """

    def __init__(self, iid, quality):
        self.iid = iid
        self.quality = quality
        self.place = []
        self.viewHistory = list()
        self.viewUid = list()
        
                
    def getID(self):
        return self.iid

    def getQuality(self):
        return self.quality
    
    def getPlace(self):
        return self.place
    
    def setPlace(self,place):
        self.place = place
        return
        
    def viewed(self,uid,eval_value):
        self.viewUid.append(uid)
        self.viewHistory.append(eval_value)
        return
        
    def getNumViews(self):
        return len(self.viewHistory)
    
    def getUpvotes(self):
        if self.viewHistory:
            return self.viewHistory.count(1)
        else:
            return 0
    
    def getDownvotes(self):
        if self.viewHistory:
            return self.viewHistory.count(2)
        else:
            return 0