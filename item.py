
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
