import numpy as np
import os
import re
import matplotlib.pyplot as plt

filepath = "../output1204/"
para = "show"
all_files = os.listdir(filepath+para+'/')


# placeholders
ps = [float("{0:.3f}".format(x)) for x in np.linspace(0.02,1,50)]
pidx = {k: v for v, k in enumerate(ps)}

mean_happy = np.zeros(len(pidx))
mean_ratio = np.zeros(len(pidx))

## READ FILES
for fi in all_files:
    print (fi)
    p = float(re.findall(r'\d+\.?\d*',fi)[0])
    p = float("{0:.3f}".format(p))
    with open(filepath+para+'/'+fi, 'r') as f:
        lines = f.readlines()
        mnhappy = lines[12][:-1].split()[-3][:]
        mnratio = lines[12][:-1].split()[-1][:-1]
        mean_happy[pidx[p]] = mnhappy
        mean_ratio[pidx[p]] = mnratio

        
## PLOT
# mean happy
fig,ax = plt.subplots()
plt.plot(ps, mean_happy, 'r*-', label='Happiness Calculated From Quality')
plt.plot(ps, mean_ratio, 'b*-', label='#Upvotes/#Views')
plt.xticks(ps)
plt.grid()
plt.minorticks_on()
plt.legend()
ax.set_xticklabels(ps,rotation=45, ha='center')
plt.title('Happiness VS. Ratio of #Items Shown')
plt.show()


