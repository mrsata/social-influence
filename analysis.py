import numpy as np
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filepath = "../output1120/"
para_pair = ['uc&nshow', 'nshow&lc', 'uc&lc']
para_pair_id = 2
all_files = os.listdir(filepath+para_pair[para_pair_id]+'/')

if para_pair_id==0:
    p1 = 'uc'
    p2 = 'nshow'
    fixed = 'tau=1,lc=1'
    p1s = np.linspace(0,3,13)
    p2s = [float("{0:.2f}".format(x)) for x in np.linspace(0.1,1,19)]
elif para_pair_id==1:
    p1 = 'nshow'
    p2 = 'lc'
    fixed = 'tau=1,uc=1'
    p1s = [float("{0:.2f}".format(x)) for x in np.linspace(0.1,1,19)]
    p2s = np.linspace(0,3,13)
elif para_pair_id==2:
    p1 = 'uc'
    p2 = 'lc'
    fixed = 'tau=1,nshow=1'
    p1s = np.linspace(0,3,13)
    p2s = np.linspace(0,3,13)

# placeholders
p1idx = {k: v for v, k in enumerate(p1s)}
p2idx = {k: v for v, k in enumerate(p2s)}
mean_happy = np.zeros((len(p1idx),len(p2idx)))
median_happy = np.zeros((len(p1idx),len(p2idx)))

## READ FILES
for fi in all_files:
    print (fi)
    para1, para2 = re.findall(r'\d+\.?\d*',fi)
    para1, para2 = float(para1), float(para2)
    with open(filepath+para_pair[para_pair_id]+'/'+fi, 'r') as f:
        lines = f.readlines()
        mn = lines[12][:-1].split()[-1][:-1]
        md = lines[13][:-1].split()[-1][:-1]
        mean_happy[p1idx[para1],p2idx[para2]] = mn
        median_happy[p1idx[para1],p2idx[para2]] = md
        
## PLOT
# mean
fig,ax = plt.subplots()
plt.imshow(mean_happy, cmap='magma', interpolation=None,aspect='auto')
ax.set_xticks(np.arange(0, len(p2s), 1))
ax.set_xticklabels(p2s,rotation=45, ha='center')
plt.xlabel(p2)
ax.set_yticks(np.arange(0, len(p1s), 1))
ax.set_yticklabels(p1s,ha='right')
plt.ylabel(p1)
plt.title('Mean Happiness ('+fixed+')')
plt.show()
plt.colorbar()

## median
#fig,ax = plt.subplots()
#plt.imshow(median_happy, cmap='magma', interpolation=None,aspect='auto')
#ax.set_xticks(np.arange(0, len(p2s), 1))
#ax.set_xticklabels(p2s,rotation=45, ha='center')
#plt.xlabel(p2)
#ax.set_yticks(np.arange(0, len(p1s), 1))
#ax.set_yticklabels(p1s,rotation=45, ha='right')
#plt.ylabel(p1)
#plt.title('Median Happiness ('+fixed+')')
#plt.show()
#plt.colorbar()


