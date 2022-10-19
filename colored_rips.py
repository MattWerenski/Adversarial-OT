import numpy as np
import numpy.linalg
import itertools
import miniball


def  pairwise_distances(points0: np.array, points1: np.array, distance='euc'):
    """
    pairwise_distances returns a matrix (np.array) of pairwise distances between 
    
    :param points0: (m,d) array of points from the first class
    :param points1: (n,d) array of points from the second class
    :param distance: Either 'euc_sq' or a distance function
    :return: (n,n) numpy array of pairwise distances
    """
    
    if distance == 'euc':
        norm_sq0 = np.square(points0).sum(axis=1)
        norm_sq1 = np.square(points1).sum(axis=1)
        dist_sq = np.add.outer(norm_sq0, norm_sq1) - 2 * points0 @ points1.T
        return np.sqrt(dist_sq)
    
    m = points0.shape[0]
    n = points1.shape[0]
    return np.array([
        distance(p1,p2) for (p1,p2) in itertools.product(points0, points1)
    ]).reshape(m,n)

def threshold_pairs(points0: np.array, points1: np.array, threshold: float, distance='euc'):
    """
    threshold_grpah returns a sparse matrix of distances below the given threshold
    
    :param points0: (m,d) array of points to consider
    :param points1: (n,d) array of points to consider
    :param threshold: positive real upper bound on the pairwise distance
    :param distance: Either 'euc_sq' or a distance function
    :return: A list of indices where the points are close enough 
        and the corresponding distances
    """
    
    # computes the pairwise distances
    distances = pairwise_distances(points0, points1, distance)
    
    # generates a binary indicator matrix
    return np.transpose(np.nonzero(distances < threshold))


def colored_rips(colored_points: np.array, threshold, max_order=3, distance='euc'):
    """
    colored_rips returns a sparse matrix of distances below the given threshold
    
    :param points: list of (n,d) array of points to consider. 
        Each outer element corresponds to a distinct color.
    :param threshold: positive real upper bound on the pairwise distance
    :param max_order: highest order to attempt to merge
    :param distance: Either 'euc_sq' or a distance function
    :return: structured dictionary of points to merge
    """
    
    k = len(colored_points)
    
    groupings = {}
    for i in range(k):
        groupings[f'({i},)'] = [tuple([i]) for i in np.arange(len(colored_points[i]), dtype=int)]
    
    
    # handles order two groups
    combos = itertools.combinations(np.arange(k),2)
    for combo in combos:
        color0 = colored_points[combo[0]]
        color1 = colored_points[combo[1]]

        pairs = threshold_pairs(color0, color1, threshold * 2)

        groupings[f'{combo}'] = [tuple(p) for p in pairs]
        
    
    # handle 3 or more, iteratively building up to higher orders
    for order in range(3,max_order+1):

        # iterate over all the unique sets of  (order) color combos
        combos = itertools.combinations(np.arange(k),order)

        for combo in combos:

            groupings[f'{combo}'] = []

            # gets the first and last (order - 1) colors
            head = combo[:-1]
            tail = combo[1:]

            # gets all the grous that were possible before adding first/last color
            head_groups = groupings[f'{head}']
            tail_groups = groupings[f'{tail}']

            # organizes all the tail_groups based on the first (order - 1) indices
            tail_groups_reorg = {}
            for tail_group in tail_groups:
                leads = tail_group[:-1]
                last = tail_group[-1]
                key = f'{leads}'
                if key in tail_groups_reorg:
                    tail_groups_reorg[key] += [last]
                else:
                    tail_groups_reorg[key] = [last]

            # iterates over all the head groups for ones that overlap tail groups
            # then sees if they are compatible

            for head_group in head_groups:

                # extracts the corresponding spatail points
                head_points = [colored_points[combo[i]][head_group[i]] for i in range(order - 1)]

                lead = head_group[0]
                lasts = head_group[1:]

                key = f'{lasts}'
                if not key in tail_groups_reorg:
                    # no groups are compatible with the current head group
                    continue

                # find the point candidates
                new_points_inds = tail_groups_reorg[key]
                new_points = [colored_points[combo[order-1]][ind] for ind in new_points_inds]

                # iterate over all the candidate groups
                for (point,ind) in zip(new_points, new_points_inds):
                    group = head_points + [point] 

                    # finds the smallest ball containing all the points
                    c,r2 = miniball.get_bounding_ball(np.array(group))

                    # if the ball is small enough, add this tuple to the grouping
                    if np.sqrt(r2) < threshold:
                        groupings[f'{combo}'] += [tuple(head_group) + tuple([ind])]
                        
    return groupings
        
        
    
    
    