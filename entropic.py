import numpy as np
from colored_rips import colored_rips

def margin_distance(a,b):
    '''
    margin_distance computes the distance between two (approximate) margins.
    This follows from the formula on page 17 of LHCJ
    
    :param a: first margin (corresponds to a prescribed margin)
    :param b: second margine (corresponds to the current multicoupling)
    :return: distance between these margins.
    '''
    
    n = a.shape[0]
    return np.dot(np.ones(n), b - a) + np.dot(a, np.log(a) - np.log(b))


def KL_divergence(projections, marginals):
    '''
    KL_divergence computes the KL divergences between marginals and projections
    
    :param projections: list of np.arrays of projection of coupling tensors
    :param marginals: prescribed marginals
    :return: np.array of KL divergences between marginals and projections
    '''
    
    K = len(projections)
    divergence = np.zeros(K)
    for k in range(K):
        margin = margin_distance(marginals[k], projections[k])
        divergence[k] = margin
    return divergence


def greedy_coordinate(projections, marginals):
    '''
    greedy_coordinate finds the coordinate of the largest KL divergence
    
    :param projections: list of np.arrays of projection of coupling tensors
    :param marginals: prescribed marginals
    :return: the coordinate of the largest KL divergence
    '''
    
    div_list = KL_divergence(projections, marginals)
    return np.argmax(div_list)


def error_threshold(projections, marginals):
    '''
    error_threshold computes the discrepancy between the marginals of the 
    current multicoupling and the given marginals
    
    :param projections: list of np.arrays of projection of coupling tensors
    :param marginals: prescribed marginals
    :return: l^1 error between marginals and projections
    '''
    
    K = len(projections)
        
    total_error = 0
    for k in range(K):        
        # computes the L1 error on this marginal and adds it to the total error 
        total_error += np.linalg.norm(marginals[k] - projections[k], ord=1)
    
    return  total_error


def full_projecting_measures(potentials, exp_cost_tensors, eta):
    
    """
    full_projecting_measures computes the sum of the projections of the couplings
    onto their marginals. It also returns the marginals of each tensor for fast
    updates in the future

    :param potentials: list of np.arrays of potential functions
    :param exp_cost_tensors: list of the exponential of cost tensors
    :param eta: entropic parameter
    :return: list of the projections and a dictionary containing the marginals
        of each of the partial couplings
    """      
    
    eta_const = np.exp(-1/eta)
    
    projections = [np.zeros(len(p)) for p in potentials]
    tensor_marginals = {}
    
    for colors in exp_cost_tensors:
        
        tensor_marginals[colors] = []
        
        # in this case we use the dense tensor approach
        if len(colors) <= 4:
            # computes the tensor product of the potentials
            exp_potential_tensor = partial_potential_tensor([potentials[i] for i in colors], eta)
            exp_cost_tensor = exp_cost_tensors[colors]
            # and multiplies by the cost tensor to get the partial coupling
            pct = exp_potential_tensor * exp_cost_tensor
            
            # computes each of the margins of the partial coupling
            sum_inds = tuple(np.arange(len(colors)))
            for i in range(len(colors)):
                sum_ind = sum_inds[:i] + sum_inds[i+1:]
                tensor_marginals[colors] += [pct.sum(sum_ind)]
                projections[colors[i]] += tensor_marginals[colors][i]
            
        # and in this case we use the sparse tensor approach
        else:
            # groups is a list of all the indices which have finite cost
            groups = exp_cost_tensors[colors]
            
            # don't take the tensor product though, we'll pull out the entries as needed
            exp_potentials = np.array([np.exp(potentials[c] / eta) for c in colors])
            inds = np.arange(len(colors))
            
            tm = [np.zeros(len(tensor_marginals[colors][i])) for i in range(len(colors))]  
            for g in groups:
                # grabs the relevant entries of the exponential potentials computes what
                # the entry of the tensor would be. Then stores it's projection
                entry = np.prod(exp_potentials[inds, g]) * eta_const
                for i in range(len(g)):
                    tm[i][g[i]] += entry
                    
            tensor_marginals[colors] = tm
                
            for i in range(len(colors)):
                projections[colors[i]] += tensor_marginals[colors][i]
                
    return projections, tensor_marginals


def projecting_measures(potentials, projections, tensor_marginals, update_index, exp_cost_tensors, eta):
    """
    projecting_measures updates the projections and tensor marginals in response
        to changing a single potential, and avoids unnecessary computation from the unchanged potentials.
        

    :param potentials: list of np.arrays of potential functions
    :param projections: list of np.arrays of the previous projections
    :param tensor_marginals: dictionary of the marginals of each of the partial coupling tensors
    :param update_index: index of the potential that w
    :param exp_cost_tensors: list of the exponential of cost tensors
    :param eta: entropic parameter
    :return: list of the projections and a dictionary containing the marginals
        of each of the partial couplings
    """      
    
    eta_const = np.exp(-1/eta)
    
    for colors in exp_cost_tensors:
        # if the class doesn't involve the index that changed we don't need to worry about it
        if not update_index in colors:
            continue
        
        # dense case
        if len(colors) <= 4:
            exp_potential_tensor = partial_potential_tensor([potentials[i] for i in colors], eta)
            exp_cost_tensor = exp_cost_tensors[colors]
            
            pct = exp_potential_tensor * exp_cost_tensor
            
            sum_inds = tuple(np.arange(len(colors)))
            for i in range(len(colors)):
                sum_ind = sum_inds[:i] + sum_inds[i+1:]
                # subtract the previous contribution from this tensor in the projection
                projections[colors[i]] -= tensor_marginals[colors][i]
                # update the marginal
                tensor_marginals[colors][i] = pct.sum(sum_ind)
                # add it back in
                projections[colors[i]] += tensor_marginals[colors][i]
        
        # sparse case
        else:
            groups = exp_cost_tensors[colors]
            exp_potentials = np.array([np.exp(potentials[c] / eta) for c in colors])
            inds = np.arange(len(colors))
            
            # subtract out the contributions from this tensor in the projection
            for i in range(len(colors)):
                projections[colors[i]] -= tensor_marginals[colors][i]
            
            # update the tensor marginals
            tm = [np.zeros(len(tensor_marginals[colors][i])) for i in range(len(colors))]  
            for g in groups:
                entry = np.prod(exp_potentials[inds, g]) * eta_const
                for i in range(len(g)):
                    tm[i][g[i]] += entry
                    
            tensor_marginals[colors] = tm
            
            # add the contribution back in
            for i in range(len(colors)):
                projections[colors[i]] += tensor_marginals[colors][i]
                
    return projections, tensor_marginals


def truncated_sinkhorn(marginals, exp_cost_tensors, eta, delta, max_order):
    '''
    truncated_sinkhorn runs a version of the sinkhorn updates which minimizes 
        redundant computations and uses sparsity
    
    :param marginals: prescribed marginals
    :param exp_cost_tensors: list of the exponential of cost tensors
    :param eta: regularization parameter
    :param delta: error threshold
    :param max_order: truncation level
    :return: potentials, projections, and the tensor_marginals for fast computation later
    '''
    
    potentials = [np.zeros(len(m)) for m in marginals]
    projections, tensor_marginals = full_projecting_measures(potentials, exp_cost_tensors, eta)
    
    error = error_threshold(projections, marginals)
    while error > delta:
        
        greedy_I = greedy_coordinate(projections, marginals)
        potentials[greedy_I] = potentials[greedy_I] + eta * np.log(marginals[greedy_I]) - eta * np.log(projections[greedy_I])
        projections, tensor_marginals = projecting_measures(potentials, projections, tensor_marginals, greedy_I, exp_cost_tensors, eta)        
        
        error = error_threshold(projections, marginals)

    return potentials, projections, tensor_marginals


def rounding_scheme(marginals, potentials, projections, tensor_marginals, exp_cost_tensors, eta):
    '''
    rounding_scheme rounds the coupling tensors to satisfy the marginal constraints. Does so
        with a minimal amount of redundant computation
        
    :param marginals: prescribed marginals
    :param potentials: list of np.arrays of potential functions
    :param projections: list of np.arrays of the previous projections
    :param tensor_marginals: dictionary of the marginals of each of the partial coupling tensors
    :param exp_cost_tensors: list of the exponential of cost tensors
    :param eta: regularization parameter
    :return: coupling_tensors satisfying the marginal constraints
    '''

    K = len(potentials)
    eta_const = np.exp(-1/eta)
    
    # iterate over each marginal, decreasing the potentials so that every marginal
    # is below its maximum allowed amount
    for k in range(K):
        # indices to marignalize over
        z_k = np.minimum( np.ones(len(projections[k])), np.divide(marginals[k], projections[k]) )
        potentials[k] = np.log(z_k) + potentials[k]
        # fast update!
        projections, tensor_marginals = projecting_measures(potentials, projections, tensor_marginals, k, exp_cost_tensors, eta)
    
    # fills in the partial couplings
    partial_coupling_tensors = {}
    for colors in exp_cost_tensors:
        # dense case
        if len(colors) <= 4:
            exp_potential_tensor = partial_potential_tensor([potentials[i] for i in colors], eta)
            exp_cost_tensor = exp_cost_tensors[colors]
            pct = exp_potential_tensor * exp_cost_tensor
            partial_coupling_tensors[colors] = pct
            continue
        
        # sparse case
        groups = exp_cost_tensors[colors]
        entries = []
        exp_potentials = np.array([np.exp(potentials[c] / eta) for c in colors])
        inds = np.arange(len(colors))
        for g in groups:
            entries += [np.prod(exp_potentials[inds, g]) * eta_const]
            
        partial_coupling_tensors[colors] = {
            'groups': groups,
            'entries': entries
        }
    
    # now go over each single interaction marginal and add any mass that is needed
    err = [np.zeros(len(marginal)) for marginal in marginals]
    for k in range(K):
        err[k] = marginals[k] - projections[k]
        partial_coupling_tensors[(k,)] += err[k]
        
    return partial_coupling_tensors

def exponential_partial_cost_tensors(data, epsilon, eta, distance, max_order, candidate='rips'):
    """
    exponential_partial_cost_tensors generates the exponential of partial cost 
    tensors for the generalized MOT. Each entry is filled in according to (2.11) in 
        https://arxiv.org/pdf/2204.12676.pdf
    which in turn uses (2.8) where the cost is assumed to be
        C(delta_x, delta_y) = { 0 if c(x,y) <= epsilon, +infty otherwise}. 
    and does so using tricks

    :param data: list of np.arrays containing the samples for each class
    :param epsilon: threshold beyond which the cost to merge points is infinity
    :param eta: entropic parameter
    :param distance: 'euclidean', 'chebyshev', or a callable distance function
    :param max_order: how large the interactions we consider should be
    :param candidate: how to check if points can be merged
    :return: tensor with entries representing the cost to merge points together
    """
    
    
    thresholds = colored_rips(data, epsilon, max_order, distance, candidate=candidate)
    
    # number of colors
    ncolors = len(data)

    # number of datapoints in each class
    sizes = [arr.shape[0] for arr in data]

    exp_cost_tensors = {}
    
    for colors in thresholds:
        # dense case
        if len(colors) <= 4:
            exp_cost_tensors[colors] = np.exp(-1/eta) * thresholds[colors]
            continue
        
        # sparse case
        exp_cost_tensors[colors] = thresholds
        
    return exp_cost_tensors


def partial_potential_tensor(potentials, eta):
    """
    partial_potential_tensor computes the tensor product of the potentials 
        using numpy to substantially speed things up
    
    :param potentials: list of np.arrays of potential functions
    :param eta: entropic parameter
    :return: tensor with entries representing the tensor product of the exponential of potentials
    """
    
    # number of colors
    ncolors = len(potentials)
    
    # this handles up to 7-way tensor products
    einsum_strings = [
        '',
        'a->a',
        'a,b->ab',
        'a,b,c->abc',
        'a,b,c,d->abcd',
        'a,b,c,d,e->abcde',
        'a,b,c,d,e,f->abcdef',
        'a,b,c,d,e,f,g->abcdefg',
        'a,b,c,d,e,f,g,h-abcdefgh'
    ]
    
    exp_potentials = [np.exp(p / eta) for p in potentials]
    # actually does the tensor producting
    fast_tensor = np.einsum(einsum_strings[ncolors], *exp_potentials)
    
    return fast_tensor


def approximate_adversarial_cost(data, marginals, epsilon, distance, delta, eta, max_order, candidate='rips'):
    '''
    approximate_adversarial_cost compute an approximate optimal cost of truncated 
        stratified MOT problem using a lot of tricks to increase speed.
    
    :param data: list of np.arrays containing the samples for each class
    :param marginals: prescribed marginals
    :param epsilon: adversarial budget
    :param distance: the choice of cost function, c
    :param delta: error threshold
    :param eta: regularization parameter
    :param max_order: truncation level
    :param candidate: method used to check if points can be merged
    :return: approximate cost: total mass of marginals - cost = adversarial risk
    '''
    
    
    
    exp_cost_tensors = exponential_partial_cost_tensors(data, epsilon, \
                                        eta, distance, max_order, candidate=candidate)
    potentials, projections, tensor_marginals = truncated_sinkhorn(marginals, exp_cost_tensors, \
                                        eta, delta, max_order)
    partial_coupling_tensors = rounding_scheme(marginals, potentials, projections, \
                                         tensor_marginals, exp_cost_tensors, eta)
    partial_cost = 0
    for interaction in partial_coupling_tensors:
        if len(interaction) <= 4:
            partial_cost += np.sum(partial_coupling_tensors[interaction])
        else:
            partial_cost += np.sum(partial_coupling_tensors[interaction]['entries'])
    
    return partial_cost