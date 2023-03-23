import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ot
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.stats import dirichlet


def generate_cost(X, Y, threshold, distance, barrier=1):
    """
    generate_cost given a pair of datapoints, gives a soft approximation to
    the indicator cost.
    
    :param X: first dataset of points
    :param Y: second dataset of points
    :param threshold: location of the soft threshold
    :param distance: distance function to compute pairwise distances
    :param barrier: parameter determining how hard the threshold should be
    :return: cost matrix approximating the indicator 
    """
    base_cost = cdist(X,Y, metric=distance)
    cost = np.exp(barrier*(base_cost - threshold))
    return cost


def solve_partial_ots(Xs, Y, Ymass, threshold, distance, barrier=1):
    """
    solve_partial_ots given a set of source distributions and a shared target
    computes the partial transport from each source to the target
    
    :param Xs: list of dataset of points
    :param Y: second dataset of points
    :param Ymass: amount of mass at each point in Y
    :param threshold: location of the soft threshold
    :param distance: distance function to compute pairwise distances
    :param barrier: parameter determining how hard the threshold should be
    :return: list of partial couplings
    """
    
    gammas = []
    for X in Xs:
        Xmass = np.ones(X.shape[0]) / X.shape[0]
        cost = generate_cost(X, Y, threshold, distance, barrier=barrier)
        gammas += [ot.partial.partial_wasserstein(Xmass, Ymass, cost, m=1, nb_dummies=10)]
        
    return gammas


def update_Y_locations(Xs, Y, gammas):
    """
    update_Y_locations moves the locations of the points in Y to the 
    centroids of the mass being assigned to them
    
    :param Xs: list of dataset points
    :param Y: second dataset of points to be updated
    :param gammas: partial couplings
    :return: updated locations as well as the updated mass assignments
    """
    
    # gets the amount of mass assigned to every point summed over
    # all of the classes
    total_assignment = np.sum(gammas, axis=(0,1))
    
    # shifts each point used in Y to the average of the points using it
    centroids = np.zeros(Y.shape)
    for (X,gamma) in zip(Xs, gammas):
        centroids += gamma.T @ X
    centroids /= total_assignment[:,np.newaxis] + 0.00000001
    
    # finds how much mass must be placed on each point in order that
    # Y dominates the transformed Xs
    dominating_mass = np.max(np.sum(gammas, axis=1), axis=0)
    
    # removes unused points 
    centroids = centroids[total_assignment > 0]
    dominating_mass = dominating_mass[total_assignment > 0]
    
    return centroids, dominating_mass


def generate_initial_Y(Xs, npoints=40):
    """
    generate_initial_Y creates a reasonable set of starting points for the 
    adversary to work from.
    
    :param Xs: list of dataset points
    :param npoints: number of support points
    """
    
    # number of classes and dimension of data
    K = len(Xs)
    d = Xs[0].shape[1]
    
    # generates random coordinates to start from
    alpha = np.ones(K)
    coords = dirichlet.rvs(alpha, size=npoints)
    
    # each point in Y is a random convex combination of real data points
    Y = np.zeros((npoints, d))
    for i in range(npoints):
        for k in range(K):
            Y[i,:] += coords[i,k] * Xs[k][np.random.randint(Xs[k].shape[0]),:]
    
    return Y

def fixed_point(Xs, threshold, distance, npoints=50, num_iters=25, barrier=1):
    """
    fixed_point given a set of measures, runs a fixed point approach to find
    the adversarial measure
    
    :param Xs: list of lists of datapoints, one for each class 
    :param threshold: location of the soft threshold
    :param distance: distance function to compute pairwise distances
    :param npoint: number of support points to start the adversary with
    :param num_iters: number of fixed point iterations to run for
    :param barrier: parameter determining how hard the threshold should be
    :return: the support, mass, and assignments for the dominating measure
    """
    
    # generates the initial setting of Y measure
    K = len(Xs)
    Y = generate_initial_Y(Xs, npoints=npoints)
    Ymass = np.ones(npoints) / npoints * K 
    
    # applies the fixed point updates 
    for i in range(num_iters):
        # solve the partial optimal transports
        gammas = solve_partial_ots(Xs, Y, Ymass, threshold, distance, barrier=barrier)
        # and uses those to update the measure
        Y,Ymass = update_Y_locations(Xs, Y, gammas)
    
    # the last gammas are useful for figuring out what the source is being moved to.
    return Y, Ymass, gammas

def violating_mass(Xs, Y, gammas, threshold, distance):
    """
    violating_mass computes how much mass from the Xs are moved too far to
    get to a Y point based on the threshold and the distance function.
    
    :param Xs: list of lists of datapoints, one for each class 
    :param Y: support of the barycenter measure
    :param threshold: maximum movement distance
    :param distance: distance function
    :return: total amount of mass across all classes moving to far
    """
    
    total_violation = 0
    for X,gamma in zip(Xs, gammas):
        base_cost = cdist(X,Y, metric=distance)
        binary_cost = base_cost > threshold
        total_violation += np.sum(binary_cost * gamma)
        
    return total_violation