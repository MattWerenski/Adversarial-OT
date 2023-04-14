import itertools
from itertools import chain, combinations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean, chebyshev


import colored_rips
import indicator_solver
import entropic

def powerset(iterable, max_order):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(max_order+1))


def create_partial_cost_tensors(data: list, epsilon: float, cost_function, max_order=3):
    """
    create_partial_cost_tensors generates the partial cost tensors for the generalized MOT. 
    Each entry is filled in according to (2.11) in https://arxiv.org/pdf/2204.12676.pdf
    which in turn uses (2.8) where the cost is assumed to be
        C(delta_x, delta_y) = { 0 if c(x,y) <= epsilon, +infty otherwise}.

    :param data: list of np.arrays containing the samples for each class
    :param epsilon: threshold beyond which the cost to merge points is infinity
    :param cost_function: 'euclidean', 'chebyshev', or a callable distance function
    :
    :return: tensor with entries representing the cost to merge points together
    """
    
    # number of classes
    nclasses = len(data)

    # number of datapoints in each class
    sizes = [arr.shape[0] for arr in data]

    # creates a set of numbers ranging from 0 to size_i for each size
    indices = [np.arange(size, dtype=int) for size in sizes]

    cost_tensors = {}
    index_tuples = {}
    
    # iterates over all types of classes up to size max_order
    for classes in powerset(np.arange(nclasses, dtype=int), max_order):
        if len(classes) == 0:
            continue
        
        used_data = [data[c] for c in classes]
        used_indices = [indices[c] for c in classes]
        cost_tensor = np.zeros([len(d) for d in used_data])
        
        for inds in itertools.product(*used_indices):            
            
            # gets all the currently used datapoints
            datapoints = [used_data[i][inds[i]] for i in range(len(classes))]
            
            # figures out the cost to merge things by looking at the datapoints
            cost = entropic.compute_cost(np.array(datapoints), epsilon, cost_function)
            cost_tensor[inds] = cost
    
        cost_tensors[str(classes)] = cost_tensor
        index_tuples[str(classes)] = classes
    
    return cost_tensors, index_tuples

def truncated_emot(data, epsilon, eta, cost_function, max_order=3):
    """
    truncated_emot solves a version of the emot where only interactions
    up to order max_order are ued

    :param data: list of np.arrays containing the samples for each class
    :param epsilon: threshold beyond which the cost to merge points is infinity
    :param eta: entropic regularization paramter
    :param cost_function: 'euclidean', 'chebyshev', or a callable distance function
    :param max_order: highest order interaction term
    :return: dictionary of groupings for each type of interaction, LL term,
        indexing dictionary
    """
    
    K = len(data)
    
    # initialize the partial cost tensors
    Bs, index_tuples = create_partial_cost_tensors(data, epsilon, cost_function, max_order)
    exp_Bs = {}
    for b in Bs:
        exp_Bs[b] = np.exp(-Bs[b] / eta)
    
    # initialize the dual potentials
    mus = [np.ones(len(d)) / len(d) for d in data]
    phis = [np.ones(len(d)) / len(d) for d in data]
    
    num_iters = 5000
    
    # outer iteration loop to obtain the potentials
    for _ in range(num_iters):
        
        # iterations over each potential
        for i in range(K):
            
            denom = np.zeros(len(phis[i]))
            
            # iterates over all the ways of grouping things together
            for group_str in exp_Bs:
                group = index_tuples[group_str]
                
                # only consider groups involving i
                if not i in group:
                    continue
                
                # handles an edge case where the group is just [i]
                if len(group) == 1:
                    denom += exp_Bs[group_str]
                    continue
                
                # sets up some stuff for tensor operations
                reshape_sizes = []
                sum_inds = []
                used_phis = []
                for j,g in enumerate(group):
                    if g == i:
                        reshape_sizes += [1]
                    else:
                        reshape_sizes += [len(data[g])]
                        sum_inds += [j]
                        used_phis += [phis[g]]
                
                # handles edge case where there is only one other potential
                if len(used_phis) == 1:
                    tensor_product = used_phis[0]
                else:
                    # computes the tensor prodcuts
                    tensor_product = np.prod(np.ix_(*used_phis), dtype=object)
                
                # and injects an extra dimension at the right place
                tensor_product.reshape(reshape_sizes)
                
                # grabs the cost tensor (already exponentiated and scaled)
                exp_cost_tensor = exp_Bs[group_str]
                
                # multiplies the tensors and sums the important axis
                denom += (tensor_product * exp_cost_tensor).sum(tuple(sum_inds))
                
            # update phi_i
            phis[i] = mus[i] / denom
            
    # potentials obtained, now rebuild the couplings
    pi_As = {}
    for group_str in Bs:
        group = index_tuples[group_str]
        
        exp_cost_tensor = exp_Bs[group_str]
        
        # handles special case of 1 potential
        if len(group) == 1:
            pi_As[group_str] = phis[group[0]] * exp_cost_tensor
            continue
            
        used_phis = [phis[g] for g in group]
        tensor_product = np.prod(np.ix_(*used_phis), dtype=object)
        
        pi_As[group_str] = tensor_product * exp_cost_tensor
        
    # computes Ll 
    Ll = 0
    # first summation (ignoring eta
    for i in range(K):
        Ll += np.sum(np.log(phis[i]) * mus[i])
        
    # second summation
    for i in range(K):
        for group_str in Bs:
            group = index_tuples[group_str]
            
            if not i in group:
                continue
                
            exp_cost_tensor = exp_Bs[group_str]
            
            # handles special case
            if len(group) == 1:
                Ll -= np.sum(exp_cost_tensor)
                continue
                
            used_phis = [phis[g] for g in group]
            tensor_product = np.prod(np.ix_(*used_phis), dtype=object)
            
            Ll -= np.sum(tensor_product * exp_cost_tensor)
    
    Ll *= eta
            
        
    return pi_As, Ll, Bs, index_tuples
                
                