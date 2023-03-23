import numpy as np
from scipy.spatial.distance import euclidean
from cvxopt import solvers
import sys
import os
sys.path.append(os.path.abspath('../'))

from colored_rips import basic_rips, colored_rips
from entropic import approximate_MOT
from indicator_solver import solve
from fixed_point import fixed_point, violating_mass

solvers.options['show_progress'] = False

m = 4
n = 12
linspace = np.arange(n)
circle = np.vstack([np.cos(linspace / n * 2 * np.pi), np.sin(linspace / n * 2 * np.pi)]).T
data = [
    circle + np.array([-1.1,-1.1]),
    circle + np.array([1.1,-1.1]),
    circle + np.array([1.1,1.1]),
    circle + np.array([-1.1,1.1]),
]


epsilon = 1.0

print("Performing eMOT")
r_ks = np.ones((m,n+1)) / (n) 
X, C = approximate_MOT(data, r_ks, epsilon, euclidean, 0.3)
print("Total Cost", (X*C).sum())

print("Performing fixed point")
Y, Ymass, gammas = fixed_point(data, epsilon, euclidean, 
                               npoints=100, num_iters=25, barrier=3)
print("Total Cost", np.sum(Ymass))
print("   Violating Mass", violating_mass(data, Y, gammas, epsilon, euclidean))

print("Performing Exact")
groupings = colored_rips(data, epsilon, m, distance=euclidean)
op, cost = solve(groupings, data, [r for r in np.ones((m,n)) / n])
print("Total Cost", cost)

