# TODO - Figure out what is making this code run slowly / stall. Not clear. 
# TODO - Check how we normalize the marginals (sum to 1 or each has mass 1?)

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from cvxopt import solvers
import sys
import os
sys.path.append(os.path.abspath('../'))

from colored_rips import colored_rips
from entropic import approximate_adversarial_cost
from indicator_solver import extract_groupings, solve

solvers.options['show_progress'] = False

m = 4
n = 12
linspace = np.arange(n)
circle = np.vstack([np.cos(linspace / n * 2 * np.pi), np.sin(linspace / n * 2 * np.pi)]).T

data = [
    circle + np.array([-0.6,-0.6]),
    circle + np.array([0.6,-0.6]),
    circle + np.array([0.6,0.6]),
    circle + np.array([-0.6,0.6]),
]
marginals = np.ones((m,n)) / (m * n)
epsilon = 1.0
distance = "euclidean"
delta = 0.00001
eta = 0.1
max_order = 4
candidate='cech'

print("Performing eMOT")

C = approximate_adversarial_cost(data, marginals, epsilon, distance, delta, \
                                     eta, max_order, candidate=candidate)
print("Total Cost", C)

print("Performing Exact")
thresholds = colored_rips(data, epsilon, max_order=max_order, \
                                distance=distance, candidate=candidate)
groupings = extract_groupings(thresholds)
op, cost = solve(groupings, data, [r for r in marginals])
print("Total Cost", cost)

