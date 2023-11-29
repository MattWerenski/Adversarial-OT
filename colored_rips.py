from itertools import combinations
import numpy as np
from scipy.spatial.distance import cdist
import miniball

def process_binary_tensor(binary_tensor, non_zero, data, threshold, distance='euclidean', candidate='rips'):
    '''
    process_binary_tensor receives a binary_tensor representing which colored points 
        can *possibly* be merged together, and validates whether or not they actually can be
        
    :param binary_tensor: 0/1 or True/False tensor which represents the possibility of merging
        (some hacks are done on this, it can be a dictionary which gets populated). 
        This is modified *in place*
    :param: non_zero: list of the non-zero entries in the binary_tensor (or a hack in some cases)
    :param data: list of lists of points that are under consideration
    :param threshold: maximum allowed distance to move
    :param distance: only matters of candidate="cech" and determines how to find the "center point"
    :param candidate: the way to check if a set of points can be merged. Options are 
        "rips" - forms the rips complex
        "cech" - forms the cech complex
        "centroid" - picks the centroid of the points as the center point
        callable - custom function which returns the center of a list of points
    :return: None, the binary_tensor is modified directly
    '''
    
    
    k = len(data)
    if candidate == 'rips':
        return
        
    if candidate == 'cech':
        if distance == 'euclidean':
            for indices in non_zero:
                points = np.array([data[i][indices[i]] for i in range(k)])

                # finds the smallest ball containing all the points
                c,r2 = miniball.get_bounding_ball(points)

                # if the ball is small enough, add this tuple to the grouping
                binary_tensor[indices] = np.sqrt(r2) <= threshold
                
        elif distance == 'chebyshev':
            for indices in non_zero:
                points = np.array([data[i][indices[i]] for i in range(k)])
                
                c = (np.max(points, axis=0) + np.min(points,axis=0)) / 2
                r = np.max(np.abs(points - c))
                binary_tensor[indices] = r <= threshold
                
        else:
            raise Exception('cech is only supported when distnance="euclidean" or "chebyshev"')
            
        return

    if candidate == 'centroid':
        for indices in non_zero:
            points = np.array([data[i][indices[i]] for i in range(k)])
            centroid = np.mean(points, axis=0)
            binary_tensor[indices] = cdist(centroid.reshape(1,-1), points, metric=distance).max() < threshold
        return

    else:
        for indices in non_zero:
            points = np.array([data[i][indices[i]] for i in range(k)])
            center = candidate(points)
            binary_tensor[indices] = cdist(center, points, metric=distance).max() < threshold

def colored_rips(data: np.array, threshold, max_order=3, distance='euclidean', candidate='rips'):
    """
    colored_rips returns a structured dictionary of the valid interactions of distances below 
        the given threshold and does so using a bunch of tricks for speed up
    
    :param points: list of (n,d) array of points to consider. 
        Each outer element corresponds to a distinct color.
    :param threshold: positive real upper bound on the pairwise distance
    :param max_order: highest order to attempt to merge
    :param distance: Either 'euclidean', 'chebyshev', or a distance function
    :param candidate: the way to check if a set of points can be merged. Options are 
        "rips" - forms the rips complex
        "cech" - forms the cech complex
        "centroid" - picks the centroid of the points as the center point
        callable - custom function which returns the center of a list of points
    :return: structured dictionary of points to merge
    """
    
    k = len(data)
    
    thresholds = {}
    # handles order 1
    for i in range(k):
        thresholds[(i,)] = np.ones(len(data[i]))
    
    if max_order == 1:
        return thresholds
    # handles order 2
    combos = combinations(np.arange(k),2)
    for combo in combos:
        color0 = data[combo[0]]
        color1 = data[combo[1]]
        dist_matrix = cdist(color0, color1, metric=distance)
        binary_tensor = dist_matrix < threshold * 2
        if np.sum(binary_tensor) == 0:
            continue 
        thresholds[combo] = binary_tensor
    
    if max_order == 2:
        return thresholds
    
    # handles order 3
    combos = combinations(np.arange(k),3)
    for combo in combos:
        
        if (not (combo[0],combo[1]) in thresholds) \
            or (not (combo[0],combo[2]) in thresholds) \
            or (not (combo[1],combo[2]) in thresholds):
            continue
        
        t0 = thresholds[(combo[0],combo[1])]
        t1 = thresholds[(combo[0],combo[2])]
        t2 = thresholds[(combo[1],combo[2])]
        
        # the binary_tensor[i,j,k] is true if it is possible that the points
        # can be merged together
        binary_tensor = np.einsum('ij,ik,jk->ijk',t0,t1,t2)
        
        cps = [data[i] for i in combo]
        non_zero = zip(*np.nonzero(binary_tensor))
        # validates which points can be merged
        process_binary_tensor(binary_tensor, non_zero, cps, threshold, distance=distance, candidate=candidate)
        
        if np.sum(binary_tensor) == 0:
            continue
            
        thresholds[combo] = binary_tensor
    if max_order == 3:
        return thresholds
    
    # handles order 4
    combos = combinations(np.arange(k), 4)
    for combo in combos:
        if (not (combo[0],combo[1],combo[2]) in thresholds) \
            or (not (combo[1],combo[2],combo[3]) in thresholds) \
            or (not (combo[0],combo[3]) in thresholds):
            continue

        t0 = thresholds[(combo[0],combo[1],combo[2])]
        t1 = thresholds[(combo[1],combo[2],combo[3])]
        t2 = thresholds[(combo[0],combo[3])]
        
        # the binary_tensor[i,j,k,;] is true if it is possible that the points
        # can be merged together
        binary_tensor = np.einsum('ijk,jkl,il->ijkl',t0,t1,t2,optimize=True)
        
        cps = [data[i] for i in combo]
        non_zero = zip(*np.nonzero(binary_tensor))
        # validates which points can be merged
        process_binary_tensor(binary_tensor, non_zero, cps, threshold, distance=distance, candidate=candidate)
        
        if np.sum(binary_tensor) == 0:
            continue

        thresholds[combo] = binary_tensor
    
    if max_order == 4:
        return thresholds
    
    # now we handle the sparse part. In this setting we don't return tensors,
    # but instead lists of indices where merging is possible
    # this is some of the hardest code to understand
    groupings = {}
    combos = combinations(np.arange(k), 4)
    for combo in combos:
        if not combo in thresholds:
            continue
        
        groupings[combo] = list(zip(*np.nonzero(thresholds[combo])))
    
    for order in range(5,max_order+1):
        # iterate over all the unique sets of  (order) color combos
        combos = combinations(np.arange(k),order)

        for combo in combos:

            # gets the first and last (order - 1) colors
            head = combo[:-1]
            tail = combo[1:]
            

            if (not head in groupings) or (not tail in groupings):
                continue

            # gets all the groupings that were possible before adding first/last color
            head_groups = groupings[head]
            tail_groups = groupings[tail]

            # organizes all the tail_groups based on the first (order - 1) indices
            tail_groups_reorg = {}
            for tail_group in tail_groups:
                leads = tail_group[:-1]
                last = tail_group[-1]
                if leads in tail_groups_reorg:
                    tail_groups_reorg[leads] += [last]
                else:
                    tail_groups_reorg[leads] = [last]
            
            non_zero = []
            
            # iterates over all the head groups for ones that overlap tail groups
            # then sees if they are compatible
            for head_group in head_groups:
                
                lasts = head_group[1:]

                if not lasts in tail_groups_reorg:
                    # no groups are compatible with the current head group
                    continue
                    
                # find the point candidates
                new_points_inds = tail_groups_reorg[lasts]
                non_zero += [head_group + (ind,) for ind in new_points_inds]
            
            valid_groupings_dict = {}
            cps = [data[i] for i in combo]
            # does a special trick for the rips complex here
            if candidate == 'rips':
                order2 = thresholds[(combo[0],combo[-1])]
                for nz in non_zero:
                    valid_groupings_dict[nz] = order2[nz[0],nz[-1]]
            else:
                process_binary_tensor(valid_groupings_dict, non_zero, cps, threshold, distance=distance, candidate=candidate)
            
            new_groups = []
            for g in valid_groupings_dict:
                if valid_groupings_dict[g]:
                    new_groups += [g]
                    
            if len(new_groups) > 0:
                groupings[combo] = new_groups
                

    for g in groupings:
        if len(g) > 4:
            thresholds[g] = groupings[g]
            
    return thresholds

