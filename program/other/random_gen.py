import networkx as nx
import numpy as np
from hamiltonian import build_sparse_hamiltonian
from diagonal import diagonalise_and_analyse

def compute_average_idos_random_graphs(N, J=1.0, k=150,
                                       num_samples=10,
                                       ws_k=4, ws_p=0.1,
                                       hyperbolic_radius=0.2):
    """
    Compute averaged IDOS over multiple samples for random graphs.
    Returns a dictionary:
        name -> (E_norm_avg, IDOS_avg)
    """

    idos_dict = {}

    # --- Watts-Strogatz ---
    idos_ws_list = []
    for _ in range(num_samples):
        G_ws = nx.watts_strogatz_graph(N, ws_k, ws_p)
        edges_ws = [(i,j,J) for i,j in G_ws.edges()]
        H = build_sparse_hamiltonian(N, edges_ws, np.zeros(N))
        E_norm, IDOS = diagonalise_and_analyse(H, k=k)
        idos_ws_list.append((E_norm, IDOS))
    
    # Average IDOS: interpolate to common grid: interpolated to the same energy grid,
    common_E = np.linspace(0,1,k)
    IDOS_ws_avg = np.zeros_like(common_E)
    for E_norm, IDOS in idos_ws_list:
        IDOS_ws_avg += np.interp(common_E, E_norm, IDOS)
    IDOS_ws_avg /= num_samples
    idos_dict['watts_strogatz_avg'] = (common_E, IDOS_ws_avg)

    # --- Hyperbolic / geometric ---
    idos_hyp_list = []
    for _ in range(num_samples):
        G_hyp = nx.random_geometric_graph(N, radius=hyperbolic_radius)
        edges_hyp = [(i,j,J) for i,j in G_hyp.edges()]
        H = build_sparse_hamiltonian(N, edges_hyp, np.zeros(N))
        E_norm, IDOS = diagonalise_and_analyse(H, k=k)
        idos_hyp_list.append((E_norm, IDOS))
    
    # Average
    IDOS_hyp_avg = np.zeros_like(common_E)
    for E_norm, IDOS in idos_hyp_list:
        IDOS_hyp_avg += np.interp(common_E, E_norm, IDOS)
    IDOS_hyp_avg /= num_samples
    idos_dict['hyperbolic_nx_avg'] = (common_E, IDOS_hyp_avg)

    return idos_dict
