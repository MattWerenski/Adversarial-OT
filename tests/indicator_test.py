import numpy as np
import scipy as sp
import scipy.stats
import itertools
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('../'))

from colored_rips import colored_rips
from indicator_solver import solve

# required settings

k = 6
max_order = 6
distance='euclidean'
threshold = 1.2

# create some basic data 

npoints = [30, 30, 30, 30, 30, 30]
centers = [(-2,2),(2,2),(6,2),(-2,-2),(2,-2),(6,-2)]

colored_points = []
for i in range(k):
    colored_points += [sp.stats.multivariate_normal.rvs(mean=centers[i], size=(npoints[i]))]

# get the groupings

groupings = colored_rips(colored_points, threshold, max_order, distance=distance)
    
# plot the points

f1 = plt.figure(1, figsize = (7,7))

for i in range(k):
    plt.scatter(colored_points[i][:,0], colored_points[i][:,1])    

# and show all the complexes of order at least 3
for o in range(3, max_order+1):
    for combo in itertools.combinations(np.arange(k), o):
        groups = groupings[f'{combo}']
        for g in groups:
            ps = [colored_points[combo[i]][g[i]] for i in range(o)]
            ps = np.array(ps + [ps[0]])
            plt.plot(ps[:,0], ps[:,1], c = 'y', alpha = 0.02 + (4-o)*0.2)
            
            
plt.legend([0,1,2,3,4,5])
plt.title('Data with Colored Complexes of Order 3+')
plt.tight_layout()

f2 = plt.figure(2, figsize = (7,7))



# place weights on the points
weights = [np.ones(n)/n for n in npoints] 
# and then actually make the groups
op, cost = solve(groupings, colored_points, weights)

print(f'total cost', cost)

legend = []
for g in op:
    points = op[g]['points']
    
    if len(points) > 0:
        legend += [g]
        plt.scatter(points[:,0], points[:,1])

plt.legend(legend)
plt.title('Optimally perturbed data')
plt.tight_layout()
plt.show()
