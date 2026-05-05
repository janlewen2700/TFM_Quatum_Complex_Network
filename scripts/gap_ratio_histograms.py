import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from utilities import set_article_style
    
def apply_unfolding_density_cutoff(evals, density_threshold=0.1, window_size=11):
    from scipy.stats import gaussian_kde
    """
    evals: raw eigenvalues
    density_threshold: float (0 to 1), keep only region where density > max_density * threshold
    """
    evals = np.sort(np.real(evals))
    
    # 1. Estimate Density using KDE
    # bw_method='silverman' usually works well for RMT spectra
    kde = gaussian_kde(evals, bw_method='silverman')
    rho = kde(evals) # Evaluate density at each eigenvalue point
    
    # 2. Filter based on max density
    rho_max = np.max(rho)
    mask = rho > (rho_max * density_threshold)
    bulk_evals = evals[mask]
    
    if len(bulk_evals) < window_size:
        return np.array([])

    # 3. Local Unfolding (your existing logic)
    unfolded = []
    for i in range(0, len(bulk_evals) - window_size + 1, window_size):
        win = bulk_evals[i:i + window_size]
        local_mean = np.mean(np.diff(win))
        if local_mean > 0:
            unfolded.extend(np.diff(win) / local_mean)
            
    return np.array(unfolded)
    
def start_end_with_vertical_cutoffs(evals, density_threshold=0.1):
    from scipy.stats import gaussian_kde
    
    # 1. Compute KDE
    kde = gaussian_kde(evals, bw_method='silverman')
    x = np.linspace(min(evals), max(evals), 1000)
    y = kde(x)
    
    # 2. Find threshold value
    thresh_value = np.max(y) * density_threshold
    
    # 3. Find crossing points (indices where density > threshold)
    mask = y >= thresh_value
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        x_start, x_end = x[indices[0]], x[indices[-1]]
    else:
        x_start, x_end = min(x), max(x)
    
    return x_start, x_end



def plot_ensemble_dos(flattened_evals, bins=100, start=1.0, end=1.0, savefig=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Density of States
    ax.hist(flattened_evals, bins=1000, density=True,
            histtype='stepfilled', color='lightgray', edgecolor='black', alpha=0.5)

    
    ax.set_title("Ensemble Density of States (Raw Eigenvalues)")
    ax.set_xlabel(r"Energy $\lambda$")
    ax.set_ylabel(r"$\rho(\lambda)$")
    
    # Highlight the edges that get removed during unfolding
    # (Visually representing remove_edges=0.1)
    q_low, q_high = np.percentile(flattened_evals, [10, 90])
    ax.axvline(q_low, color='red', linestyle=':', label='Bulk Cutoff')
    ax.axvline(q_high, color='red', linestyle=':')
    
    ax.axvline(start, color='orange', linestyle=':', label='KDE cutoff')
    ax.axvline(end, color='orange', linestyle=':')
    
    
    
    ax.legend()
    if savefig:
        plt.savefig(savefig)
    plt.show()



def plot_aggregated_spectra(prefix, folder="../spectra", s_max=4, ax=None, show_xlabel=True,  show_ylabel=True, show_xticklabels=True, show_yticklabels=True):
    
    
    # 1. Find all matching files
    search_path = os.path.join(folder, f"{prefix}*.npz")
    files = glob.glob(search_path)
    
    if not files:
        print("No files found matching that prefix.")
        return

    all_unfolded_spacings = []
    all_raw_evals = []
    
    # 2. Process each file
    for f in files:
        data = np.load(f)
        evals = data["evals"]
        all_raw_evals.append(evals)
        
        # Use your existing unfolding logic here
        #unfolded = apply_unfolding(evals, remove_edges=0.1, window_size=11)
        unfolded = apply_unfolding_density_cutoff(evals, density_threshold=0.2, window_size=11)
        all_unfolded_spacings.append(unfolded)

       
       
    flattened_raw = np.concatenate(all_raw_evals)
    st, en = start_end_with_vertical_cutoffs(flattened_raw, density_threshold=0.2)
    
    total_gaps = sum(len(s) for s in all_unfolded_spacings)
    bins = int(np.sqrt(total_gaps / len(files)))
    

    bin_edges = np.linspace(0, s_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate hist for each realization
    hist_runs = [np.histogram(s, bins=bin_edges, density=True)[0] for s in all_unfolded_spacings]
    avg_hist = np.mean(hist_runs, axis=0)
    std_err = np.std(hist_runs, axis=0) / np.sqrt(len(files))

    if ax is None:
        fig, ax = plt.subplots()
    
    
    # Theoretical Curves
    s_fine = np.linspace(0, s_max, 200)
    ax.plot(s_fine, (np.pi*s_fine/2)*np.exp(-np.pi*s_fine**2/4), 'k-', label='GOE', alpha=0.7, lw=1)
    ax.plot(s_fine, np.exp(-s_fine), 'k--', label='Poisson', alpha=0.7, lw=1)

    # "No filling" histogram: use drawstyle='steps-mid'
    # This looks like a clean histogram outline
    ax.errorbar(bin_centers, avg_hist, yerr=std_err, fmt='none', ecolor='k', capsize=2)
    ax.plot(bin_centers, avg_hist, drawstyle='steps-mid', color='blue', linewidth=1.5)

    if show_xlabel:
        ax.set_xlabel(r"$s$")
    else:
        ax.set_xlabel("")
    
    if show_ylabel:
        ax.set_ylabel(r"$P(s)$")
    else:
        ax.set_ylabel("")
    
    ax.tick_params(axis='x', labelbottom=show_xticklabels, bottom=True)
    ax.tick_params(axis='y', labelleft=show_yticklabels, bottom=True)

        
    
    ax.set_xlim(0, s_max)
    ax.set_ylim(0, 1.1) # P(s) usually stays below 1.0 for s > 0
    
    return ax, flattened_raw, st, en

def plot_two_regimes(param1, param2, folder="SD", savefig=None):

    set_article_style()

    # Adjust figure size for a stacked column plot (Width, Total Height)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.375, 3.5), sharex=True)

    # Adjust spacing so they touch
    fig.subplots_adjust(hspace=0.03)

    # Top plot: No xlabel, no numbers
    _, evals1, st1, en1 =plot_aggregated_spectra(param1, folder=folder, ax=ax1, show_xlabel=False, show_ylabel=False, show_xticklabels=False, show_yticklabels=False)
    #ax1.set_title("Weak Interaction", loc='right', fontsize=10, pad=-15) # Internal label

    # Bottom plot: Has everything
    _, evals2, st2, en2 =plot_aggregated_spectra(param2, folder=folder, ax=ax2, show_xlabel=True, show_xticklabels=True)
    ax2.tick_params(labelbottom=True)
    #ax2.set_title("Strong Interaction", loc='right', fontsize=10, pad=-15)

    # Save
    if savefig:
        fig.savefig(savefig)
        
    plt.show()
    
    plot_ensemble_dos(evals1, bins=1000, start=st1, end=en1, savefig="../figures/evals_small_world.pdf")
    plot_ensemble_dos(evals2, bins=1000, start=st2, end=en2, savefig="../figures/evals_large_world.pdf")
    
    plt.close()
    


#We use this function to plot an aggregated spectra. The number of bins is prop to the number of eigenvalues approx. The parameters prefix is what we are looking for in the files.

plot_two_regimes("SD_N=4000_D=1_B=1.21_G=2.", "SD_N=4000_D=1_B=10.01_G=8.", folder = "../spectra/SD", savefig="../figures/gap/SD_spacings_regimes.pdf")


