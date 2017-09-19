
'''
May need:
    viewHistory
    upvoteHistory
'''

import random
#from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from item import Item
from user import User
import xlwt
from functions import *
import numpy as np


# Create excel sheet to store records
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Records")

num_item = 20
num_user = 30
items = list();
users = list();
quality_range = range(1,6)  # quality of 1~5
viewHistory = np.zeros([num_item,num_user+1])

# Initialization of items (and save to excel)
idx = 0
sheet1.write(0, idx, "Item ID")
sheet1.write(0, idx+1, "Quality")
for i in range(0,num_item):
    q = random.choice(quality_range)
    items.append(Item(i,q))
    viewHistory[i,0] = 0
    sheet1.write(i+1, idx, i)
    sheet1.write(i+1, idx+1, q)
idx += 3

# Initialization of users
for i in range(0,num_user):
    users.append(User(i))

# Random order
items_rdm = list(items)
random.shuffle(items_rdm)
for i,itm in enumerate(items_rdm): 
    itm.setPlace(i)
print ("Items in random order: ", [(i.getID(),i.getQuality(),i.getPlace()) for i in items_rdm])

displayCurItems(items_rdm,idx,0,sheet1,"","(start)")
idx += 7

# Users start to view items...
for u in users:
    # Display items in random order to user
    random.shuffle(items_rdm)
    for i,itm in enumerate(items_rdm): 
        itm.setPlace(i)  # reset the place values for items
        # if being viewed
        if random.random()>(i+1)/(num_item+1):
            # whether to upvote/downvote/do nothing
            rdm1 = random.random()
            rdm2 = random.random()
            qlt = itm.getQuality()
            a = qlt/5*0.8
            if rdm1<a or rdm2<0.2:
                itm.viewed(u.getID(),1)  # upvote
            elif rdm2>0.8:
                itm.viewed(u.getID(),2)  # downvote
            else:
                itm.viewed(u.getID(),0)  # do nothing
        viewHistory[itm.getID(),u.getID()+1] = itm.getUpvotes()-itm.getDownvotes()
    displayCurItems(items_rdm,idx,u.getID(),sheet1,"")
    idx += 7

book.save("TrialRecords.xls")
plt.imshow(viewHistory, cmap=plt.cm.Blues, interpolation='nearest')
plt.yticks(range(0,num_item), [itm.getQuality() for itm in items])
plt.colorbar()
plt.show()
plt.savefig('Trial.png')


#Blues = plt.get_cmap('Blues')
#bg = Image.new('RGB', (512, 512), (255,255,255))
#info = ImageDraw.Draw(bg)
#info.text((10,10),"Item ID, Quality, Views, Upvotes, Downvotes",(100,100,100))
#for i,itm in enumerate(items_rdm): 
##    print ([int(x*255) for x in Blues(itm.getQuality()/5)[:3]])
#    colors = tuple([int(x*255) for x in Blues(itm.getQuality()/5)[:3]])
#    info.text((10,20+10*i),str(itm.getID())+" "+str(itm.getQuality())+" "+str(itm.getNumViews())+" "+str(itm.getUpvotes())+" "+str(itm.getDownvotes()),colors)
#
#bg.save("temp.png")
#bg.show()