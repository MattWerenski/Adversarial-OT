import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from cvxopt import solvers
import sys
import os
sys.path.append(os.path.abspath('../'))

from colored_rips import basic_rips, colored_rips
from entropic import approximate_MOT
from indicator_solver import solve
from fixed_point import incentive_fixed_point, violating_mass
from truncated import truncated_emot

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

#circle = np.vstack([np.cos(linspace / n * 2 * np.pi), np.sin(linspace / n * 2 * np.pi)]).T
#data = [
#    circle + np.array([-1.1,-1.1]),
#    circle + np.array([1.1,-1.1]),
#    circle + np.array([1.1,1.1]),
#    circle + np.array([-1.1,1.1]),
#]

#x1 = multivariate_normal.rvs(size=(12,2)) / 4
#x2 = multivariate_normal.rvs(size=(12,2)) / 4
#x3 = multivariate_normal.rvs(size=(12,2)) / 4
#x4 = multivariate_normal.rvs(size=(12,2)) / 4

#x1 += np.array([-1.5,-0.5])
#x2 += np.array([-1.5,0.5])
#x3 += np.array([1.5,-0.5])
#x4 += np.array([1.5,0.5])

#data = [ x1, x2, x3, x4 ]

epsilon = 0.3
eta = 1

print("Performing eMOT")
r_ks = np.ones((m,n+1)) / (n) 
X, C = approximate_MOT(data, r_ks, epsilon, euclidean, 0.3)
print("Total Cost", (X*C).sum())

print("Performing truncated eMOT")
pi_As, Ll, Bs, index_tuples = truncated_emot(data, epsilon, eta, euclidean, max_order=3)
c = 0
for p in pi_As:
    c += np.sum(pi_As[p] * Bs[p])
print("Total Cost", c)

print("Performing fixed point")
Y, Ymass, gammas, _ = incentive_fixed_point(data, epsilon, euclidean, 
                               npoints=10, num_iters=25, barrier=3)
print("Total Cost", np.sum(Ymass))
print("   Violating Mass", violating_mass(data, Y, gammas, epsilon, euclidean))

print("Performing Exact")
groupings = colored_rips(data, epsilon, m, distance=euclidean)
op, cost = solve(groupings, data, [r for r in np.ones((m,n)) / n])
print("Total Cost", cost)

