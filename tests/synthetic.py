import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import tqdm

sys.path.append(os.path.abspath('../'))
from colored_rips import colored_rips
import entropic
import indicator_solver

# create some basic data 
npoints = [50, 50, 50, 50, 50, 50]
centers = [(-2,2),(2,2),(6,2),(-2,-2),(2,-2),(6,-2)]

data = []
for i in range(len(npoints)):
    data += [sp.stats.multivariate_normal.rvs(mean=centers[i], size=(npoints[i]))]
    
marginals = [np.ones(n) / np.sum(npoints) for n in npoints]

thresholds = np.arange(16) / 2.5 
distance = 'euclidean'
candidate = 'rips'
delta = 0.0001
etas = np.arange(11) / 10
max_order = 3



pbar = tqdm.tqdm(total=len(etas) * len(thresholds))

result_set = []
# outer loop is for the regularization parameter
for i in range(etas.shape[0]):
    eta = etas[i]
    loss = np.zeros(thresholds.shape)
    # inner loop is for epsilon / the threshold parameter
    for j in range(thresholds.shape[0]):
        # when eta = 0.0 we do the LP approach
        if eta == 0.0:
            groupings = indicator_solver.extract_groupings(
                colored_rips(data, thresholds[j], max_order=max_order, 
                                                distance=distance, candidate=candidate)
            )
            loss[j] = indicator_solver.solve(groupings, data, marginals,
                                             return_mu_As=False, metric=distance)
        else:
            loss[j] = entropic.approximate_adversarial_cost(data, marginals, thresholds[j], 
                            distance, delta, eta, max_order)
        pbar.update(1)
        
    result = {
        'distance': distance,
        'candidate': candidate,
        'delta': delta,
        'max_order': max_order,
        'eta': eta,
        'thresholds': list(thresholds),
        'loss': list(loss)
    }

    result_set += [result]
    
pbar.close()

cmap = mpl.cm.get_cmap('inferno')
for r in result_set:
    plt.plot(r['thresholds'], r['loss'], color=cmap(r['eta']))
    
plt.plot([0,6],[1/max_order, 1/max_order],'--')
plt.show()
