
import itertools
import numpy as np
from scipy.spatial.distance import cdist
import torch
import miniball

import colored_rips
import indicator_solver

def find_components(A: np.array):
    """
    find_components given a boolean matrix representing the adjacency matrix
    returns a list of indices of that are in each component.

    :param A: square boolean np.array where A_ij = True if i is connected to j
    :return: a list of list of indices for each component.
    """

    # tracks where we have been so far
    visited = np.zeros(A.shape[0])
    # will contain all the components
    components = []
    
    # this will evaluate to false once every index has been assigned a component
    while np.sum(visited) < A.shape[0]:
        # find the first index not yet visited 
        start = np.argmin(visited)

        # set the search to start there
        queue = [start]
        # and add this index to the componnet
        comp = [start]
        # mark that one as visited
        visited[start] = 1

        # performs a breadth first search of the graph building up the component
        # when the queue is empty we have gone over the whole component
        while len(queue) > 0:
            # pop the first index off the queue
            current = queue[0]
            queue = queue[1:]
            
            # visited * (-1) + 1 = 0 if that index was already visited and 1 otherwise
            # performing this multiplication masks the neighbors of current if they
            # have already been visited.
            unvistied_neighbors = A[current] * (visited * (-1) + 1)
            indices = np.nonzero(unvistied_neighbors)[0].tolist()
            
            # adds all new indices to the queue
            queue += indices
            # adds all new indices to the component
            comp += indices
            # and marks them as visited
            visited += unvistied_neighbors
            
        # adds the completed component to the list
        components += [comp]

    # returns all of the components
    return components


def compute_cost(datapoints: list, epsilon: float, cost_function):
    """
    compute_cost given a set of datapoints, figures out how the points can be
    merged together while staying under the budget epsilon. The UNNORMALIZED
    cost is returned (all points are treated as if they have mass 1) 
    (TODO: eventually allow non-uniform mass)

    :param datapoints: list of points to merge
    :param epsilon: threshold beyond which the cost to merge points is infinity
    :param cost_function: 'euclidean', 'chebyshev' or a callable distance function
    :return: the cost to merge
    """
    
    if cost_function == 'euclidean':
        metric = euclidean
    elif cost_function == 'chebyshev':
        metric = chebyshev
    else:
        metric = cost_function
    
    ndatapoints = len(datapoints)
    
    # handle the three easiest cases separately
    if ndatapoints == 0:
        return 0
    if ndatapoints == 1:
        return 1
    if ndatapoints == 2:
        return 1 if metric(datapoints[0],datapoints[1]) <= 2 * epsilon else 2
    
    cost_matrix = cdist(datapoints, datapoints, metric=cost_function)
    # makes dealing with distances from i to i less annoying
    cost_matrix = cost_matrix + (np.eye(ndatapoints) * 10 * epsilon) 
    
    if ndatapoints == 3:
        # no points can be merged
        if np.min(cost_matrix) > 2*epsilon:
            return 3
        
        # all three points can be merged to a single point
        
        if cost_function == 'euclidean':
            centroid, _ = miniball.get_bounding_ball(datapoints)
        elif cost_function == 'chebyshev':
            centroid = (np.min(datapoints, axis=0) + np.max(datapoints, axis=0)) / 2
        else:
            centroid = np.mean(datapoints, axis=0)
        
        if np.max(cdist([centroid], datapoints, metric=cost_function)) <= epsilon:
            return 1
        
        # all three pairs can be merged but not to a singly point
        if np.max(np.min(cost_matrix, axis=0)) <= 2*epsilon:
            return 1.5
        
        # otherwise only pairs can be merged and the best cost is 2
        return 2
    
    # If we are using 4+ then things can get complicated. 

    # To speed things up I partition the data_points according to if they
    # can even be possibly connected based on the 2 * epsilon heuristic.
    connectable_indices = find_components(cost_matrix <= 2 * epsilon)
    
    total_cost = 0
    for component in connectable_indices:
        if len(component) < 4:
            # basic case handled above
            total_cost += compute_cost(datapoints[component,:], epsilon, cost_function)
        else:
            # more difficult case where we need to be precise
            k = len(component)

            # gets all the ways we can group points together
            groupings = colored_rips.basic_rips(datapoints[component,:], epsilon, cost_function)
            
            total_cost += indicator_solver.compute_merging_cost(k, groupings)
            
    return total_cost


def create_cost_tensor(data: list, epsilon: float, cost_function):
    """
    create_cost_tensor generates the cost tensor for the generalized MOT. 
    Each entry is filled in according to (2.11) in https://arxiv.org/pdf/2204.12676.pdf
    which in turn uses (2.8) where the cost is assumed to be
        C(delta_x, delta_y) = { 0 if c(x,y) <= epsilon, +infty otherwise}.

    :param data: list of np.arrays containing the samples for each class
    :param epsilon: threshold beyond which the cost to merge points is infinity
    :param cost_function: 'euclidean', 'chebyshev' or a callable distance function
    :return: tensor with entries representing the cost to merge points together
    """

    # number of classes
    nclasses = len(data)

    # number of datapoints in each class, and adds one for the ghost
    sizes = [arr.shape[0]+1 for arr in data]

    # creates a set of numbers ranging from 0 to size_i for each size
    indices = [np.arange(size, dtype=int) for size in sizes]

    # creates a big empty cost tensor
    cost_tensor = np.zeros(np.prod(sizes)).reshape(sizes)
    
    for inds in itertools.product(*indices):            
        # obtains all the indices which are not the end indices
        datapoints = [data[i][inds[i]] if inds[i]+1 != sizes[i] else None for i in range(nclasses)]
        datapoints = [dp for dp in datapoints if dp is not None]
    
        # figures out the cost to merge things by looking at the datapoints
        cost = compute_cost(np.array(datapoints), epsilon, cost_function)
        cost_tensor[inds] = cost
        

    return cost_tensor

def error_threshold(approx_margins, r_ks):
    '''
    error_threshold computes the discrepancy between the marginals of the 
    current multicoupling and the given marginals
    
    :param B: tensor current approximate multicoupling
    :param r_ks: prescribed marginals
    :return: error in the marginals, this is equation (17) in LHCJ
    '''
    
    m = approx_margins.shape[0]
        
    total_error = 0
    # iterate over each margin
    for k in range(m):        
        # computes the L1 error on this marginal and adds it to the total error 
        total_error += np.linalg.norm(r_ks[k] - approx_margins[k], ord=1)
    
    print(f'Total error {total_error}', end='\r')
    return  total_error


def coupling_from_potentials(beta, cost_tensor, eta):
    '''
    coupling_from_potentials computes the current (approximate) coupling
    which is induced by the dual potentials beta and the cost tensor
    in LHCJ this is on page 8 and referred to using B
    
    :param beta: 2D np.array where each row is a dual potential
    :param cost_tensor: tensor of merging costs
    :param eta: regularization parameter
    :return: approximate multicoupling induced by the potentials
    '''
    
    (m,n) = beta.shape
    
    # computes the outer sum of the potentials. After this is complete
    # outer_sum[i0,i1,...,im-1] = beta[0,i0] + beta[1,i1] + ... + beta[m-1,im-1] 
    shape = np.ones(m, dtype=int)
    shape[0] = n
    outer_sum = beta[0].reshape(tuple(shape))
    for k in range(1,m):
        shape = np.ones(m, dtype=int)
        shape[k] = n
        outer_sum = outer_sum + beta[k].reshape(tuple(shape))
        
    return np.exp(outer_sum - cost_tensor / eta)


def compute_margins(mc):
    '''
    compute_margins given a multicoupling as a tensor, reutrns the margins
    by summing over all but one index in each possible way.
    
    :param mc: tensor representing the multicoupling
    :return: array of marginals of the multicoupling
    '''
    
    # extracts the number of margins and the size from the multicoupling
    m = len(mc.shape)
    n = mc.shape[0]
    
    approx_margins = torch.zeros((m,n))
    # iterates over the axes and sums along all but one.
    for k in range(m):
        sum_inds = np.arange(m-1, dtype=int) 
        sum_inds[k:] += 1
        approx_margins[k,:] = torch.sum(mc, tuple(sum_inds))
        
    return approx_margins


def margin_distance(a,b):
    '''
    margin_distance computes the distance between two (approximate) margins.
    This follows from the formula on page 17 of LHCJ
    
    :param a: first margin (corresponds to a prescribed margin)
    :param b: second margine (corresponds to the current multicoupling)
    :return: distance between these margins.
    '''
    
    n = a.shape[0]
    return np.dot(np.ones(n), b - a) + np.dot(a, np.log(a / b))
    
    
    
def multi_sinkhorn(cost_tensor, eta, r_ks, epsilon_prime):
    '''
    multi_sinkhorn performs algorithm 1 in LHCJ which take a cost tensor
    and performs sinkhorn iterations until the error is small enough
    
    :param cost_tensor: tensor where each entry is the cost to couple points
    :param eta: regularization parameter
    :param r_ks: 2D np.array where the kth row is the prescribed marginal of the
        kth distribution being coupled (these are r_k in the paper)
    :param epsilon_prime: error threshold
    :return: approximate multicoupling of the data
    '''
    
    (m,n) = r_ks.shape
    
    beta = torch.zeros(r_ks.shape)
    B = coupling_from_potentials(beta, cost_tensor, eta)
    approx_margins = compute_margins(B)
    while error_threshold(approx_margins, r_ks) > epsilon_prime:
        
        # finds the distances between current margins and prescribed margins
        rhos = np.zeros(m)
        for k in range(m):
            # finds the distance between the current margin and prescribed margin
            rhos[k] = margin_distance(r_ks[k], approx_margins[k])
        
        # finds the worst margin which is to be used for the update
        K = np.argmax(rhos)
        
        # performs the update on the potential for that margin
        beta[K] = beta[K] + np.log(r_ks[K] / approx_margins[K])
        
        # updates the multicoupling and the margins
        B = coupling_from_potentials(beta, cost_tensor, eta)
        approx_margins = compute_margins(B)
        
    return B 

def round_solution(X, r_ks):
    '''
    round_solution performs algorithm 2 LHCJ which takes an approximate coupling
    from multi_sinkhorn and rounds it appropriately to get an exact coupling
    
    :param X: tensor representing the coupling
    :param r_ks: 2D np.array where the kth row is the prescribed marginal of the
        kth distribution being coupled (these are r_k in the paper)
    :return: a perturbed version of X which is exactly a coupling
    '''
    
    (m,n) = r_ks.shape
    X_k = X
    
    # iterate fixing one margin at a time
    for k in range(m):
        # indices to marignalize over
        sum_inds = np.arange(m-1, dtype=int)
        sum_inds[k:] += 1 
        
        # shape for tensor multiplication
        shape = np.ones(m, dtype=int)
        shape[k] = n
        
        # finds z_k as defined in alg 2.
        z_k = np.minimum(np.ones(n), np.divide(r_ks[k], np.array(torch.sum(X_k, tuple(sum_inds)))))
        
        # performs the update on X_k but all at once using a tensor multiplication trick
        X_k = X_k * z_k.reshape(tuple(shape))
    
    # computes the err_k for each index at the end. (error in margin k)
    err_ks = torch.zeros(r_ks.shape)
    approx_margins = compute_margins(X_k)
    err_ks = r_ks - approx_margins
        
    # finds the final coupling
    err_1_norm = torch.sum(torch.abs(err_ks[0,:]))
    
    # computes the outer product of the errors to apply
    shape = np.ones(m, dtype=int)
    shape[0] = n
    error_product = err_ks[0].reshape(tuple(shape))
    for k in range(1,m):
        shape = np.ones(m, dtype=int)
        shape[k] = n
    
        error_product = error_product * err_ks[k].reshape(tuple(shape))
    
    # final formula for Y
    Y = X_k + error_product / torch.pow(err_1_norm, m - 1)
    
    return Y


def approximate_MOT(data, r_ks, epsilon, cost_function, delta):
    '''
    approximate_MOT solves for the approximate MOT multicoupling using 
    algorithm 3 in LHCJ.
    
    :param data: list of data points
    :param r_ks: list of weights for each datapoint
        (currently this must be uniform)
    :param epsilon: tolerance under which points can be combined
    :param cost_function: method of computing distances between points
    :param delta: accuracy parameter (this is epsilon in LHCJ)
    :return: a multicoupling of the margins
    '''
    
    (m,n) = r_ks.shape
    
    print('start')
    
    # creates the cost tensor for the problem
    cost_tensor = create_cost_tensor(data, epsilon, cost_function)
    
    print('cost tensor complete')
    
    eta = delta / (2 * m * np.log(n+1))
    epsilon_prime = delta / (8 * torch.max(torch.Tensor(cost_tensor)))
    
    r_ks_tilde = (1 - epsilon_prime / (4 * m)) * r_ks + (epsilon_prime / (4 * m * (n+1)))
    
    print('parameter set')
    
    # step 2 of Alg. 3, get an approximate multicoupling
    X_tilde = multi_sinkhorn(cost_tensor, eta, r_ks_tilde, epsilon_prime)
    
    print('\nmulti sinkhorn complete')
    
    # step 3 of Alg. 3, perform the rounding step
    X_hat = round_solution(X_tilde, r_ks_tilde)
    
    print('rounding complete')
    print('done')
    
    return X_hat, cost_tensor