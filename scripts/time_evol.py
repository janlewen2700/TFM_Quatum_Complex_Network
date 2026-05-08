import glob
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from utilities import set_article_style

def get_hyperbolic_distance(r1, theta1, r2, theta2):
    """Computes the exact hyperbolic distance between two points."""
    # cosh(d) = cosh(r1)cosh(r2) - sinh(r1)sinh(r2)cos(theta1-theta2)
    # Clipping for numerical stability
    arg = np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos(theta1 - theta2)
    return np.arccosh(np.maximum(1.0, arg))

def compute_geometric_nature_standalone(coords, p, HypR, threshold=0.05):
    idx_max = np.argmax(p)
    p_max = p[idx_max]
    r_max, th_max = coords.iloc[idx_max][['radius', 'theta']]
    
    mask = p >= (threshold * p_max)
    if np.sum(mask) < 2: return 0.0
    
    w = p[mask] / np.sum(p[mask])
    r, th = coords['radius'].values[mask], coords['theta'].values[mask]
    
    # Using your existing get_hyperbolic_distance
    d_ij = get_hyperbolic_distance(r, th, r_max, th_max)
    return np.sum(d_ij * w) / HypR
    
    
def get_distances_from_start(start_idx, r, theta):
    """
    Computes distances from a specific node to all other nodes.
    start_idx: The index of your initial node |100...0>
    r: 1D array of radial coordinates for ALL nodes
    theta: 1D array of angular coordinates for ALL nodes
    """
    # Extract coordinates for the start node
    r_s = r[start_idx]
    theta_s = theta[start_idx]
    
    # Vectorized calculation against the entire arrays of r and theta
    arg = np.cosh(r_s) * np.cosh(r) - np.sinh(r_s) * np.sinh(r) * np.cos(theta_s - theta)
    
    # Clip and return 1D array of distances
    return np.arccosh(np.maximum(1.0, arg))

def get_ipr(amplitudes):
    """Computes IPR = sum(|psi|^4)"""
    return np.sum(np.abs(amplitudes)**4)

def plot_quantum_spread(file_path, start_idx, times):
    """
    start_node_idx: The index of the initial node |100...0>
    times: List of times T to evaluate [1, 10, 100, ..., np.inf]
    """
    data = np.load(file_path)
    
    evals = data['eigenvalues']
    evecs = data['eigenvectors']
    r = data['radius']
    th = data['theta']
    
    # 1. Get distances from the starting node to all others
    dists = get_distances_from_start(start_idx, r, th)
    max_d = np.max(dists)
    min_d = np.min(dists)
    norm_dists = 1-dists / max_d  # Scale from 0 to 1
    
    max_time=1e8
    times1 = np.logspace(0, np.log10(max_time), 100)
    ipr_values = []
    
    # 3. Calculate evolution and IPR
    c_n = evecs[start_idx, :]
    
    for t in times1:
        amp_t = evecs @ (c_n * np.exp(-1j * evals * t))
        ipr_values.append(get_ipr(amp_t))
        
    # 4. Calculate Time-Averaged IPR (The limit as T -> infinity)
    # For a non-degenerate system: IPR_avg = sum_j (sum_n |phi_n(start)|^2 |phi_n(j)|^2)^2
    # But often people just look at the IPR of the time-averaged distribution itself:
    c_n_sq = evecs[start_idx, :]**2
    p_avg = np.dot(evecs**2, c_n_sq)
    ipr_inf = np.sum(p_avg**2)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: IPR Evolution
    ax1.plot(times1, ipr_values, label='Instantaneous IPR(t)', color='blue')
    ax1.axhline(y=ipr_inf, color='red', linestyle='--', label='Time-Averaged Limit')
    ax1.set_xscale('log')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('IPR')
    ax1.set_title('Quantum State Participation over Time')
    ax1.legend()
    
    # Define distance bins for smoothing the plot (0 to 1)
    bins = np.linspace(0, 1, 100)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for T in times:
        if T == np.inf:
            # Time-averaged: P_avg = sum_n |phi_n(start)|^2 * |phi_n(j)|^2
            c_n_sq = evecs[start_idx, :]**2
            probs = np.dot(evecs**2, c_n_sq)
            label = r"Time-Averaged ($\infty$)"
        else:
            # Instantaneous: |psi(T)|^2
            c_n = evecs[start_idx, :]
            # Using @ for matrix-vector multiplication
            amplitudes_T = evecs @ (c_n * np.exp(-1j * evals * T))
            probs = np.abs(amplitudes_T)**2
            label = f"T = {T}"

        # 2. Bin probabilities by normalized distance
        # digitize assigns each node to a bin based on its distance
        bin_indices = np.digitize(norm_dists, bins) - 1
        binned_probs = np.zeros(len(bin_centers))
        
        for i in range(len(norm_dists)):
            if 0 <= bin_indices[i] < len(binned_probs):
                binned_probs[bin_indices[i]] += probs[i]
        
        ax2.plot(bin_centers, binned_probs, label=label, alpha=0.8)

    ax2.set_xlabel("Normalized Hyperbolic Distance (0=Start, 1=Furthest)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Quantum State Exploration over Geometric Space")
    ax2.set_yscale('log')
    ax2.set_xscale('log') # Log scale helps see the exponential decay in localized states
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    
    
    plt.show()






def plot_full_analysis(probs, radius, theta, sizes, savefig=None):
    
    set_article_style()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    
    max_r = max(radius)*1.02
    
    ax.scatter(theta, radius, c=probs, s=sizes, cmap='plasma_r', alpha=0.8, edgecolors='none')


    ax.set_ylim(0, max_r)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if savefig:
        plt.savefig(savefig)
    
    
    plt.show()
    plt.close()



def analyze_quantum_exploration(file_path, start_node_idx=0):
    # Load the compressed data
    data = np.load(file_path)
    
    evals = data['eigenvalues']
    evecs = data['eigenvectors']
    r = data['radius']
    th = data['theta']
    degrees = data['degrees']
    
    max_id=np.argmax(degrees)
    print(max_id)
    
    # --- Option A: The Stationary (Time-Averaged) Limit ---
    # Formula: P_avg(j) = sum_n |psi_n(start)|^2 * |psi_n(j)|^2
    probs_sq = evecs**2
    # Row-column multiplication: sum over the energy index (columns)
    #exploration_map = np.dot(probs_sq, probs_sq[start_node_idx, :])
    
    exploration_map = np.dot(probs_sq, probs_sq[start_node_idx, :])
    
    sizes = sizes = 2 * (1 + (degrees / degrees.mean()))
    
    # --- Option B: Time Evolution at a specific time T ---
    # psi(T) = sum_n ( e^{-i*lambda_n*T} * psi_n(start) * |psi_n> )
    # This would give you the "snapshot" at time T
    
    return exploration_map, r, th, sizes


def get_time_evolution_probs(evecs, evals, start_node_idx, t):
    # c_n are the weights of the initial node in the energy basis
    c_n = evecs[start_node_idx, :]
    
    # Evolve the amplitudes with the complex phase
    # np.exp(-1j * evals * t) is a vector of phases
    amplitudes_t = np.dot(evecs, c_n * np.exp(-1j * evals * t))
    
    # Probability is the absolute square
    return np.abs(amplitudes_t)**2

def create_quantum_gif(file_path, start_node_idx, duration=10, frames=100, output="../figures/gif/quantum.gif"):
    from matplotlib.animation import FuncAnimation


    data = np.load(file_path)
    evals, evecs = data['eigenvalues'], data['eigenvectors']
    r, th, deg = data['radius'], data['theta'], data['degrees']
    
    sizes = 2 * (1 + (deg / deg.mean())) # Use your degree-based sizing
    times = np.linspace(0, duration, frames)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    
    # Initialize the scatter plot
    scat = ax.scatter(th, r, c=np.zeros_like(th), s=sizes, cmap='plasma_r', alpha=0.8, zorder=1, vmin=0, vmax=0.005)
    
    start_point = ax.scatter(th[start_node_idx], r[start_node_idx],
               color='red', s=sizes[start_node_idx],
               edgecolors='white', linewidth=1.5, zorder=2)
    ax.set_ylim(0, max(r)*1.02)
    ax.set_xticks([]); ax.set_yticks([])


    def update(frame):
        t = times[frame]
        probs = get_time_evolution_probs(evecs, evals, start_node_idx, t)
        scat.set_array(probs) # Update colors[cite: 1]
        
        if frame > 0:
            start_point.set_alpha(0)
        else:
            start_point.set_alpha(1)
        return (scat,)

    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(output, writer='pillow', fps=20)
    plt.close()








file_SW="../spectra/SPECTRA_20260505_181533_269505_b1.21_g2.21_N3990.npz"
file_LW="../spectra/SPECTRA_20260505_182010_821024_b10.01_g8.01_N4000.npz"
file_SW_HOMO="../spectra/SPECTRA_20260506_173215_817043_b1.21_g8.01_N4000.npz"


idx_array = np.random.randint(0, 4000, size=1)
times=[1,10,100,1000,10000, np.inf]

for idx in idx_array:
    plot_quantum_spread(file_SW, idx, times)


plot_quantum_spread(file_SW, 101, times)


#exploration, radius, theta, sizes = analyze_quantum_exploration(file_SW, start_node_idx=idx)
#plot_full_analysis(exploration, radius, theta, sizes, savefig="../figures/gif/time_average_SW.pdf")

#exploration, radius, theta, sizes = analyze_quantum_exploration(file_SW, start_node_idx=101)
#plot_full_analysis(exploration, radius, theta, sizes, savefig="../figures/gif/time_average_SW_HUB.pdf")

#exploration, radius, theta, sizes = analyze_quantum_exploration(file_SW_HOMO, start_node_idx=idx)
#plot_full_analysis(exploration, radius, theta, sizes, savefig="../figures/gif/time_average_SW_homo.pdf")



#create_quantum_gif(file_SW, 101, duration=100, frames=100, output="../figures/gif/SW_hub.gif")

#create_quantum_gif(file_SW, idx, duration=100, frames=100, output="../figures/gif/SW_random.gif")

#create_quantum_gif(file_LW, 1903, duration=100, frames=100, output="../figures/gif/LW_hub.gif")

#create_quantum_gif(file_SW_HOMO, 152, duration=100, frames=100, output="../figures/gif/SW_homo_hub.gif")
