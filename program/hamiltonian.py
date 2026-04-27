from scipy.sparse import coo_matrix, diags
import numpy as np

def build_sparse_hamiltonian(N, edges, J, epsilon=None):

    if epsilon is None:
        epsilon = np.zeros(N)
    
    row = []
    col = []
    data = []
    
    # this changes absolutely the hamiltonian, but i am looking to see what happens
    #epsilon = np.random.uniform(-1e-2, 1e-2, size=N)

    # hopping terms
    for i, j, Jij in edges:
        row.extend([i, j])
        col.extend([j, i])
        data.extend([-Jij, -Jij])  # tight-binding sign

    # build hopping matrix
    H = coo_matrix((data, (row, col)), shape=(N, N))

    # add onsite
    H = H + diags(epsilon)

    return H.tocsr()
