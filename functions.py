#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:36:50 2017

@author: xiranliu
"""

#from item import Item
#from user import User
#import xlwt

def writeHeaderHelper(sheet,r,c):
    sheet.write(r, c, "No.")
    sheet.write(r, c+1, "ID")
    sheet.write(r, c+2, "Quality")
    sheet.write(r, c+3, "Number of Views")
    sheet.write(r, c+4, "Upvotes")
    sheet.write(r, c+5, "Downvotes")

def writeContentHelper(sheet,r,c,itm):
    sheet.write(r, c+1, itm.getID())
    sheet.write(r, c+2, itm.getQuality())
    sheet.write(r, c+3, itm.getNumViews())
    sheet.write(r, c+4, itm.getUpvotes())
    sheet.write(r, c+5, itm.getDownvotes())
    
# display current iterms in specific order and output to excel
def displayCurItems(items,idx,run_num,sheet,order="",comment=""):
    items_desq = sorted(items, key=lambda x: x.quality, reverse=True)
    items_ascid = sorted(items, key=lambda x: x.iid, reverse=False)
    for i,itm in enumerate(items_desq): 
        itm.setPlace(i)
    print ("(Item ID, Quality, Views, Upvotes, Downvotes)")
    if order=="des_q":
        print ("Current items ",comment," in descending quality order: ")
        print ([(i.getID(),i.getQuality(),i.getNumViews(),i.getUpvotes(),i.getDownvotes()) for i in items_desq])
    else:
        print ("Current items ",comment," in ID order: ")
        print ([(i.getID(),i.getQuality(),i.getNumViews(),i.getUpvotes(),i.getDownvotes()) for i in items_ascid])
    # Save to Excel
    sheet.write(0, idx, run_num)
    sheet.write(0, idx+1, "ID Order")
    writeHeaderHelper(sheet,1,idx)
    for i,itm in enumerate(items_ascid):
        sheet.write(i+2, idx, i)
        writeContentHelper(sheet,i+2,idx,itm)
    j = len(items_ascid)+3
    sheet.write(j, idx, run_num)
    sheet.write(j, idx+1, "Quality Order")
    writeHeaderHelper(sheet,j+1,idx)
    for i,itm in enumerate(items_desq):
        sheet.write(j+i+2, idx, i)
        writeContentHelper(sheet,j+i+2,idx,itm)