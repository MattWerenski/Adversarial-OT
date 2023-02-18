import numpy as np
from colored_rips import basic_rips
from cvxopt import matrix, solvers, spdiag, spmatrix
import miniball

def compute_merging_cost(k, groupings):
    """
    compute_meging_cost given a set of datapoints each from a different class 
    and mass one finds the exact optimal way of merging the points together.

    :param k: number of points being considered
    :param groupings: output of basic_rips, a dictionary where the keys are the
        valid ways to merge things together
    :return: the cost to merge
    """
    
    ngroups = len(groupings)
    
    # builds a group-point incidence matrix
    incidence_matrix = np.zeros((k,ngroups))
    i = 0
    for g in groupings:
        # converts the string '(0, 1, 2)' to the array [0, 1, 2]
        group = np.fromstring(g[1:-1], dtype=int, sep=',')

        incidence_matrix[group, i] = 1
        i += 1

    sum_matrix = matrix(incidence_matrix)
    sum_vector = matrix(np.ones(k))

    # creates a matrix-vector pair for enforcing non-negativity constraints
    non_neg_matrix = matrix(np.diag(np.ones(ngroups)*-1))
    non_neg_vector = matrix(0, (ngroups, 1), tc='d')
    
    # vector for the inner product in the LP
    objective_vector = matrix(1.0, (ngroups,1), tc='d')
    
    # solves the linear program for the assignments
    # for details see https://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.lp
    sol = solvers.lp(objective_vector, non_neg_matrix, non_neg_vector, 
        A=sum_matrix, b=sum_vector)
    
    return sol['primal objective'] 
    

def make_sparse_sum_constraints(groupings, npoints):
    """
    generates the equality constraint matrix for the LP below. 
    
    :param groupings: colored vietoris-rips groups, this is output by colored_rips 
    :param npoints: list of number of points for each color
    :return: A sparse matrix for enforcing the equality constraints in the LP
    """
    # helper function for ease of indexing rows across colors
    start_inds = np.cumsum(npoints) - npoints[0]
    def h_index(g,i):
        return start_inds[g] + i
    
    # for generating a sparse all-1's matrix you need two arrays
    # one for the row, and one for columns indices.
    # each row corresponds to a single point
    # and each column corresponds to a colored complex.
    row_inds = []
    col_inds = []
    
    # the matrix is filled column-by-column
    current_col = 0
    
    # each "grouping" corresponds to a group of colors being fused 
    for g in groupings:
        # converts the string '(0, 1, 2)' to the array [0, 1, 2]
        groups = np.fromstring(g[1:-1], dtype=int, sep=',')
        
        # iterates over all given complexes
        complexes = groupings[g]
        for comp in complexes:
            # the row indices correspond to the points in the complex
            row_inds += [h_index(g,i) for (g,i) in zip(groups,comp)]
            
            # each row index needs a corresponding column index
            col_inds += [current_col] * len(groups)
            
            # move on to the next complex
            current_col += 1    
    
    # creates a sparse matrix compatible with cvxopt for fast LP solving
    return spmatrix(1.0, # every non-zero entry is a 1.0
                    np.array(row_inds, dtype=int), 
                    np.array(col_inds,dtype=int), 
                    tc='d') # d is required by the library


def solve(groupings, colored_points, weights):
    """
    solve returns the optimal measures mu_A based on the groupings and weights
    
    :param groupings: colored vietoris-rips groups, this is output by colored_rips
    :param colored_points: list of arrays of points, each outer element is one color
    :param weights: list of vectors representing the amount of mass at each point 
    :return: A dictionary of optimal point placements and weights
    """
    
    npoints = [len(w) for w in weights]
    
    # creates a matrix-vector pair for enforcing equality constraints
    sum_matrix = make_sparse_sum_constraints(groupings, npoints)
    sum_vector = matrix(np.concatenate(weights))
    
    # creates a matrix-vector pair for enforcing non-negativity constraints
    non_neg_matrix = spdiag([-1]*sum_matrix.size[1])
    non_neg_vector = matrix(0, (sum_matrix.size[1], 1), tc='d')
    
    # vector for the inner product in the LP
    objective_vector = matrix(1.0, (sum_matrix.size[1],1), tc='d')
    
    # solves the linear program for the assignments
    # for details see https://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.lp
    sol = solvers.lp(objective_vector, non_neg_matrix, non_neg_vector, 
        A=sum_matrix, b=sum_vector) 
    
    assignments = np.array(sol['x']).squeeze()
    cost = np.sum(assignments)
    
    mu_As = {}
    
    # index keeps track of the appropriate location in the assignments vector
    index = 0
    # iterates over the subsets of {1,...,K} and constructs mu_A
    for g in groupings:
        # converts the string '(0, 1, 2)' to the array [0, 1, 2]
        groups = np.fromstring(g[1:-1], dtype=int, sep=',')
        
        points = []
        opt_weights = []
        
        # iterates over the complexes and finds both the fuse point and the
        # amount of mass going to that point
        complexes = groupings[g]
        for c in complexes:
            # truncates sicne cvxopt can have some low precision at times.
            if assignments[index] < 0.00001:
                index += 1
                continue

            # adds the mass
            opt_weights += [assignments[index]]

            # uses miniball to find the location of the support
            # 1. collects the point locations from their indices
            ps = np.array([colored_points[k][i] for (k,i) in zip(groups, c)])
            # 2. find the correct miniball placement
            (p,r) = miniball.get_bounding_ball(ps)
            
            points += [p]
            index += 1

        points = np.array(points)
        opt_weights = np.array(opt_weights)

        mu_As[g] = { 'points': points, 'weights': opt_weights }
    
    return mu_As, cost
    
    