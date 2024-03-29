import numpy as np
import scipy as sp
import scipy.stats
import itertools
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('../'))

from colored_rips import colored_rips

# required settings

k = 6
max_order = 4
distance='euclidean'
threshold = 1.5

# create some basic data 

npoints = [30, 30, 30, 30, 30, 30]
centers = [(-2,-2),(-2,2),(2,2),(2,-2),(6,2),(6,-2)]

colored_points = []
for i in range(k):
    colored_points += [sp.stats.multivariate_normal.rvs(mean=centers[i], size=(npoints[i]))]

# get the groupings

groupings = colored_rips(colored_points, threshold, max_order, distance=distance)
    
# plot the points
    
for i in range(k):
    plt.scatter(colored_points[i][:,0], colored_points[i][:,1])    


# and show all the complexes of order at least 4
for combo in groupings:
    if len(combo) < 4:
        continue

    groups = groupings[combo]
    for g in zip(*np.nonzero(groups)):
        ps = [colored_points[combo[i]][g[i]] for i in range(len(combo))]
        ps = np.array(ps + [ps[0]])
        plt.plot(ps[:,0], ps[:,1], c = 'y', alpha = 0.04)
            
plt.show()

