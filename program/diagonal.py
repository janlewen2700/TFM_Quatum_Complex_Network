import numpy as np
from scipy.sparse.linalg import eigsh
from plots import plot_idos_vs_normalised_energy
from hamiltonian import build_sparse_hamiltonian


def diagonalise_and_analyse(H, k=150):
    """
    Diagonalise Hamiltonian H and compute normalised eigenvalues and IDOS.
    Returns:
        E_norm : normalised energies in [0,1]
        IDOS   : integrated density of states
    """
    from scipy.sparse.linalg import eigsh

    N = H.shape[0]
    k = min(k, N-2)
    #evals = eigsh(H, k=k, which='SA', return_eigenvectors=False)
    evals = np.linalg.eigvalsh(H.toarray())
    evals = np.sort(evals)

    # normalise energy to [0,1]
    E_min, E_max = evals.min(), evals.max()
    print("evals:")
    print(evals.min(), evals.max())
    E_norm = (evals - E_min) / (E_max - E_min)

    # IDOS
    #IDOS = np.arange(1, k+1) / k
    m=len(evals)
    IDOS = np.arange(1, m+1)/m

    return E_norm, IDOS

def compute_idos_for_graphs(graphs, k=150):
    """
    For each graph in 'graphs', compute normalised energies and IDOS.
    Returns a dict: name -> (E_norm, IDOS)
    """
    idos_dict = {}

    for name, (edges, epsilon) in graphs.items():
        N = len(epsilon)
        H = build_sparse_hamiltonian(N, edges, epsilon)
        E_norm, IDOS, evals = diagonalise_and_analyse(H, k=k)
        idos_dict[name] = (E_norm, IDOS, evals)

    return idos_dict


def compute_load_eigenvalues(self, folder="../spectra", force=False, save=True):
    from scipy.sparse.linalg import eigsh
    import os

    path = self._spectrum_filename(folder)
        
    # In-memory cache
    if self._eigenvalues is not None and not force:
        return self._eigenvalues, self._ground_energy, self._ground_state

    # Load from disk if exists
    if os.path.exists(path) and not force:
        print(f"Loading spectrum from {path}")
        data = np.load(path)

        self._eigenvalues = data["evals"]
        self._ground_energy = data["ground_energy"]
        self._ground_state = data["ground_state"]

        return self._eigenvalues, self._ground_energy, self._ground_state

    # Otherwise compute
    print("Diagonalising Hamiltonian...")

    H = self.build_hamiltonian()

    # --- full spectrum ---
    evals = np.linalg.eigvalsh(H.toarray())
    evals = np.sort(evals)
    

    # --- ground state via sparse ---
    ground_energy, ground_state = eigsh(H, k=1, which='SA')

    ground_energy = ground_energy[0]
    ground_state = ground_state[:, 0]
    
    #to be modified at the correct time so i can have some saying from the top.
    prob = np.abs(ground_state)**2
    threshold = np.percentile(prob, 90)
    
    self.vec0_metrics = compute_ground_state_metrics(self, ground_state, threshold, use_prob=True)

    # Cache in memory
    self._eigenvalues = evals
    self._ground_energy = ground_energy
    self._ground_state = ground_state

    # Save everything
    if save:
        np.savez(path, evals=evals, ground_energy=ground_energy, ground_state=ground_state)

    #print(f"Saved spectrum to {path}")

    return evals, ground_energy, ground_state



def compute_ground_state_metrics(self, vec0, threshold, use_prob=True):
    """
    Compute localization/connectivity metrics for the ground state.

    Parameters
    ----------
    vec0 : np.ndarray
        Ground-state eigenvector.
    threshold : float
        Threshold applied to |vec0|^2 if use_prob=True, otherwise to |vec0|.
    use_prob : bool, default=True
        Whether to threshold on probability |vec0|^2.

    Returns
    -------
    metrics : dict
        Dictionary with a compact set of observables.
    """
    import numpy as np
    import networkx as nx

    if not hasattr(self, "edges"):
        raise AttributeError(f"{self.label} has no attribute 'edges'.")

    # --- amplitudes / probabilities ---
    amp = np.abs(vec0)
    prob = amp**2
    values = prob if use_prob else amp

    N = len(vec0)

    # --- participation metrics ---
    ipr = np.sum(prob**2)
    participation_number = 1.0 / ipr if ipr > 0 else 0.0
    participation_fraction = participation_number / N if N > 0 else 0.0

    # --- selected nodes ---
    selected = np.where(values > threshold)[0]
    n_selected = len(selected)

    # --- build full graph ---
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for edge in self.edges:
        if len(edge) >= 2:
            i, j = edge[:2]
            if len(edge) == 3:
                G.add_edge(i, j, weight=edge[2])
            else:
                G.add_edge(i, j)

    E_total = G.number_of_edges()

    # --- induced subgraph on selected nodes ---
    H = G.subgraph(selected).copy()
    E_sub = H.number_of_edges()

    metrics = {
        "label": getattr(self, "label", "graph"),
        "threshold": float(threshold),
        "use_prob": bool(use_prob),
        "N": int(N),
        "E_total": int(E_total),

        # localization
        "ipr": float(ipr),
        "participation_number": float(participation_number),
        "participation_fraction": float(participation_fraction),

        # thresholded support
        "n_selected": int(n_selected),
        "fraction_selected_nodes": float(n_selected / N) if N > 0 else 0.0,
        "E_sub": int(E_sub),
    }

    # --- empty case ---
    if n_selected == 0:
        metrics.update({
            "avg_internal_degree": 0.0,
            "density_subgraph": 0.0,
            "n_components": 0,
            "largest_component_fraction": 0.0,
            "avg_degree_retention": 0.0,
        })
        return metrics

    # --- average internal degree ---
    deg_sub = dict(H.degree())
    metrics["avg_internal_degree"] = (
        float(np.mean(list(deg_sub.values()))) if deg_sub else 0.0
    )

    # --- density of induced subgraph ---
    metrics["density_subgraph"] = float(nx.density(H)) if n_selected > 1 else 0.0

    # --- connected components ---
    comps = list(nx.connected_components(H))
    n_components = len(comps)
    largest_comp_size = max((len(c) for c in comps), default=0)

    metrics["n_components"] = int(n_components)
    metrics["largest_component_fraction"] = (
        float(largest_comp_size / n_selected) if n_selected > 0 else 0.0
    )

    # --- degree retention ---
    # average fraction of each selected node's original degree that stays inside H
    deg_full = dict(G.degree(selected))
    ratios = []
    for i in selected:
        k_full = deg_full.get(i, 0)
        k_sub = deg_sub.get(i, 0)
        ratios.append(k_sub / k_full if k_full > 0 else 0.0)

    metrics["avg_degree_retention"] = float(np.mean(ratios)) if ratios else 0.0
    
    #deg probability map that should tell us how the ground state is localised.
    all_degrees = np.array([G.degree(i) for i in range(N)])
    
    # Pre-aggregate: sum probabilities for each unique degree
    deg_prob_sum = {}
    deg_node_count = {}

    for k, p in zip(all_degrees, prob):
        deg_prob_sum[k] = deg_prob_sum.get(k, 0.0) + p
        deg_node_count[k] = deg_node_count.get(k, 0) + 1

    # Store both in your metrics
    metrics["deg_prob_map"] = deg_prob_sum
    metrics["deg_node_count"] = deg_node_count
    
    return metrics





"""
Thuis is the gap ratio part, could be moved to a different place so i can later better understand what each part is doing. Also, at some point, consider taking everyting into a csv, instead of generating and analysing each time.

"""

def compute_gap_ratio(self, remove_edges=0):

    evals, _, _ = self.compute_eigenvalues()
        
    evals = np.sort(np.real(evals))

    # Optional: remove edges of spectrum
    if remove_edges > 0:
        num_to_remove = int(len(evals) * remove_edges)
        evals = evals[num_to_remove : -num_to_remove]

    # Level spacings
    deltas = np.diff(evals)

    # Adjacent ratios
    r = np.minimum(deltas[1:], deltas[:-1]) / np.maximum(deltas[1:], deltas[:-1])

    r_mean = np.mean(r)
        
    print(len(np.unique(evals)))
    print(len(evals))
        
    return r, r_mean, deltas
    
def compute_gap_ratio_unfolded(self, remove_edges=0.1, window_size=11):

    evals, _, _ = self.compute_eigenvalues()
    evals = np.sort(np.real(evals))

    # ---- (ii) Remove edges ----
    if remove_edges > 0:
        num_to_remove = int(len(evals) * remove_edges)
        bulk_evals = evals[num_to_remove:-num_to_remove].copy()
        
    raw_spacings = np.diff(bulk_evals)
    r = np.minimum(raw_spacings[1:], raw_spacings[:-1]) / np.maximum(raw_spacings[1:], raw_spacings[:-1])
    r_mean = np.mean(r)
    self.r_mean=r_mean
    
    self.gap0 = (evals[1]-evals[0])/np.mean(raw_spacings)

    unfolded_spacings = []

    # ---- (iii) Process in local windows ----
    for start in range(0, len(bulk_evals) - window_size + 1, window_size):
        window = evals[start:start + window_size]

        # Compute spacings in window
        spacings = np.diff(window)

        # ---- (iv) Local mean spacing ----
        local_mean = np.mean(spacings)

        # ---- (v) Rescale spacings ----
        if local_mean > 0:
            unfolded_spacings.extend(spacings / local_mean)

    unfolded_spacings = np.array(unfolded_spacings)

    self.dyson_beta, _ = estimate_dyson_beta(unfolded_spacings, s_max=0.6)
    
    #self.fit_spacing_distribution_from_spacings(unfolded_spacings)

    return r, r_mean, unfolded_spacings
    
    
def estimate_dyson_beta(unfolded_spacings, s_max=0.6):
    """
    Estimates the Dyson exponent beta by fitting log(P(s)) vs log(s)
    for small spacings (s < s_max).
    """
    # 1. Create the distribution histogram
    # Density=True is critical for P(s) to be normalized
    N = len(unfolded_spacings)
    counts, bin_edges = np.histogram(unfolded_spacings, bins=int(np.sqrt(N)), range=(0, 3), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Isolate the small-s regime for fitting
    # We avoid s=0 to stay in the log-domain
    mask = (bin_centers > 0.02) & (bin_centers <= s_max)
    s_vals = bin_centers[mask]
    p_vals = counts[mask]

    # Filter out any bins with zero counts
    valid = p_vals > 0
    s_vals, p_vals = s_vals[valid], p_vals[valid]

    if len(s_vals) < 3:
        return np.nan, "Insufficient data in small-s regime."

    # 3. Linear regression on log-log data
    # log(P(s)) = beta * log(s) + log(C)
    log_s = np.log(s_vals)
    log_p = np.log(p_vals)
    
    beta_fit, intercept = np.polyfit(log_s, log_p, 1)

    return beta_fit, (s_vals, p_vals)
    
    
def fit_spacing_distribution_from_spacings(self, unfolded_spacings, bins=40, s_max=3.0):
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.special import gamma

    unfolded_spacings = np.asarray(unfolded_spacings)

    if len(unfolded_spacings) < 10:
        print("Not enough unfolded spacings to fit distribution.")
        self.brody_beta = np.nan
        self.brody_beta_err = np.nan
        self.spacing_fit_label = None
        self.spacing_fit_error_poisson = np.nan
        self.spacing_fit_error_wigner = np.nan
        return np.nan, np.nan, None

    hist, edges = np.histogram(
        unfolded_spacings,
        bins=bins,
        range=(0, s_max),
        density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    def poisson_pdf(s):
        return np.exp(-s)

    def wigner_dyson_pdf(s):
        return 0.5 * np.pi * s * np.exp(-np.pi * s**2 / 4.0)

    def brody_pdf(s, beta):
        a = gamma((beta + 2.0) / (beta + 1.0)) ** (beta + 1.0)
        return (beta + 1.0) * a * s**beta * np.exp(-a * s**(beta + 1.0))

    beta = np.nan
    beta_err = np.nan

    try:
        popt, pcov = curve_fit(
            brody_pdf,
            centers,
            hist,
            p0=[0.5],
            bounds=(0.0, 1.0)
        )
        beta = popt[0]
        beta_err = np.sqrt(np.diag(pcov))[0]
    except RuntimeError:
        print("Brody fit failed.")

    err_poisson = np.mean((hist - poisson_pdf(centers))**2)
    err_wigner = np.mean((hist - wigner_dyson_pdf(centers))**2)
    label = "Poisson" if err_poisson < err_wigner else "Wigner-Dyson"

    self.unfolded_spacings = unfolded_spacings
    self.brody_beta = beta
    self.brody_beta_err = beta_err
    self.spacing_fit_label = label
    self.spacing_fit_error_poisson = err_poisson
    self.spacing_fit_error_wigner = err_wigner

    return beta, beta_err, label
    

