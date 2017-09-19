
class User(object):
    """Abstract base class for users

    A user is an agent that can decide whether to view and to upvote the item.

    Property:
        uid
    Function:
        view
        eval

    """

    def __init__(self, uid):
        self.uid = uid
        
    def getID(self):
        return self.uid

    def view():
        pass

    def eval():
        pass
