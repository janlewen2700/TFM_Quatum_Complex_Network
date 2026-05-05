import glob
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from utilities import set_article_style


def theoretical_p_k(k_range, gamma_val, k_avg):
    from scipy.integrate import quad
    from scipy.special import gammaln
    
    kappa_0 = ((gamma_val - 2) / (gamma_val - 1)) * k_avg
    
    pk_list = []
    threshold = 100 # Where we switch from integral to pure power law
    
    # Pre-calculate log prefactor for the integral
    log_prefactor_const = np.log(gamma_val - 1) + (gamma_val - 1) * np.log(kappa_0)

    for k in k_range:
        if k < threshold:
            # INTEGRAL REGIME (Robust for the 'hump')
            integrand = lambda kappa: np.exp(
                k * np.log(kappa) - kappa - gammaln(k + 1) +
                log_prefactor_const - gamma_val * np.log(kappa)
            )
            res, _ = quad(integrand, kappa_0, np.inf, limit=100)
            pk_list.append(res)
        else:
            # ASYMPTOTIC REGIME (The 'Tail')
            # P(k) -> rho(k)
            pk_tail = (gamma_val - 1) * (kappa_0**(gamma_val - 1)) * (k**-gamma_val)
            pk_list.append(pk_tail)
            
    return np.array(pk_list)

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

def get_aggregated_data_from_log(log_csv_path, vec_folder, target_state="vec0", target_beta=1.21, target_gamma=2.21):
    """
    Reads the CSV log and loads associated binary eigenvectors.
    target_state: 'vec0', 'vec1', 'bulk', or 'max'
    """
    df = pd.read_csv(log_csv_path)
    
    df = df[df['N']>3600].copy()
    
    mask = np.isclose(df['gamma'], target_gamma) & np.isclose(df['beta'], target_beta)
    filtered_df = df[mask].reset_index(drop=True)
    
    if filtered_df.empty:
        print(f"Warning: No data found for gamma={target_gamma}, beta={target_beta}")
        return None
    
    all_degrees = [] # We still need to load these from the npz or CSV
    all_probs = []
    all_dispersions = []
    all_ipr = []
    snapshot = None

    k=0
    for idx, row in filtered_df.iterrows():
        # 1. Path to the binary file
        vec_path = os.path.join(vec_folder, f"{row['run_id']}.npz")
        if not os.path.exists(vec_path): continue
        
        # 2. Load the binary data
        data = np.load(vec_path)
        
        # get the target state
        current_vec = data.get(target_state)
        if current_vec is None: continue
        
        
        current_probs = current_vec**2
        
        # 3. Get metrics

        ipr = np.sum(current_probs**2)
        coords_tmp = pd.DataFrame({'radius': data['radius'], 'theta': data['theta']})
        disp = compute_geometric_nature_standalone(coords_tmp, current_probs, row['HypR'])

        all_probs.extend(current_probs.astype(np.float32))
        all_ipr.append(ipr)
        all_dispersions.append(disp)
        all_degrees.extend(data["degrees"])
        
        # 4. Handle Snapshot (Row 2)
        if snapshot is None and k==2:
            degree_values=data["degrees"]
            # We recreate the format Row 2 expects
            snap_coords = pd.DataFrame({
                'vertex': [f"v{i}" for i in range(len(data['radius']))],
                'radius': data['radius'],
                'theta': data['theta']
            })
            
            snap_coords[['radius', 'theta']] = snap_coords[['radius', 'theta']].astype(np.float32)
            # Sizing logic based on probs
            mean_p = np.mean(current_probs)
            sizes = sizes = 2 * (1 + (degree_values / degree_values.mean()))

            snapshot = (snap_coords, current_probs.astype(np.float32), sizes.astype(np.float32))
        k+=1

    return {
        "deg": np.array(all_degrees),
        "prob": np.array(all_probs),
        "dist": np.array(all_dispersions),
        "ipr": np.array(all_ipr),
        "snap": snapshot
    }

def plot_full_analysis(bundle1, bundle2, savefig=None):
    from scipy.stats import binned_statistic
    
    set_article_style()
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = fig.add_gridspec(3, 2, hspace=0.2, wspace=0.14)
    
    bundles = [bundle1, bundle2]
    sub_labels = ["(a)", "(c)", "(e)", "(b)", "(d)", "(f)"]
    label_idx = 0
    gammas = [2.21, 8.01]
    k_avg = 15
    
    snap1 = bundle1["snap"]
    snap2 = bundle2["snap"]


    for i, data in enumerate(bundles):
        degs, probs = data["deg"], data["prob"]
        coords, c_prob, sizes = data["snap"]
        
        max_r = coords['radius'].max()*1.02

        # --- ROW 1: Structural Localization (Snapshot Only) ---
        ax1 = fig.add_subplot(gs[0, i], projection='polar')
        ax1.scatter(coords['theta'], coords['radius'],
                    c=c_prob, s=sizes,
                    cmap='plasma_r', alpha=0.8, edgecolors='none')
        
        ax1.set_ylim(0, max_r)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.text(-0.37, 1.03, sub_labels[label_idx], transform=ax1.transAxes, fontweight='bold'); label_idx += 1


        # --- ROW 1: Structural Localization (Snapshot Only) ---
        ax2 = fig.add_subplot(gs[1, i])
        pos = ax2.get_position()
        ax2.set_position([pos.x0, pos.y0 + 0.02, pos.width, pos.height])
        
        # 1. Define Logarithmic Bins for Degree k
        # From min degree to max degree, e.g., 10 bins per decade
        bins = np.logspace(np.log10(degs.min()), np.log10(degs.max()), 20)
        bin_centers = (bins[:-1] * bins[1:])**0.5 # Geometric centers for log-scale
        
        # 2. Calculate Mean and Standard Deviation/Error
        bin_means, _, _ = binned_statistic(degs, probs, statistic='mean', bins=bins)
        bin_std, _, _ = binned_statistic(degs, probs, statistic='std', bins=bins)
        bin_count, _, _ = binned_statistic(degs, probs, statistic='count', bins=bins)
        
        # Standard Error = std / sqrt(n)
        bin_sem = bin_std / np.sqrt(bin_count)
        
        # 3. Plot with Error Bars
        # We only plot bins that actually have nodes in them
        valid = bin_count > 0
        ax2.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_sem[valid],
                     fmt='o', color='k', ecolor='k', capsize=3,
                     markersize=1, elinewidth=1)
    

        # 4. Overlay Theory
        k_theory = np.logspace(np.log10(degs.min()), np.log10(degs.max()), 100)
        pk = theoretical_p_k(k_theory, gammas[i], k_avg)
        ax2.plot(k_theory, pk, 'r--', alpha=0.7)
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(1e-6,1.5e-1)
        
        if i==1:
            ax2.tick_params(axis='both', which='both', bottom=True, left=True, right=False, top=False, labelleft=False)
        else:
            ax2.set_xlabel(r"$k$", labelpad=-5)
            ax2.set_ylabel(r"$\langle |\psi|^2 \rangle$")
            ax2.tick_params(axis='both', which='both', left=True, labelleft=True, top=False, right=False)
            
        
        ax2.text(-0.05, 1.05, sub_labels[label_idx], transform=ax2.transAxes, fontweight='bold'); label_idx += 1



    # --- ROW 3: Geometric vs. Structural Localization ---
    ax_ipr = fig.add_subplot(gs[2, 0])
    pos = ax_ipr.get_position()
    ax_ipr.set_position([pos.x0, pos.y0, pos.width, pos.height])
    ax_geo = fig.add_subplot(gs[2, 1])
    pos = ax_geo.get_position()
    ax_geo.set_position([pos.x0, pos.y0, pos.width, pos.height])
    
    
    # Unified configuration
    colors = ['#000000', '#808080'] # Teal and Dark Orchid
    labels = [rf'$USW$', rf'$NSW$']
    linestyles = ['--', '-']
    metrics = ["ipr", "dist"]
    axes = [ax_ipr, ax_geo]
    x_labels = ["IPR", "Geometric Spread"]

    for j, (metric, ax, x_lab) in enumerate(zip(metrics, axes, x_labels)):
        for i, (bundle, color, label) in enumerate(zip(bundles, colors, labels)):
            data_points = bundle[metric]
            
            # 1. Professional Histogram (Step line, no fill)
            ax.hist(data_points, bins=25, color=color, histtype='step',
                    linewidth=1.2, ls=linestyles[i], density=True, alpha=0.8, label=label)
            
            # 2. Smooth Fit (KDE is more professional for non-Gaussian physics data)
            #kde = gaussian_kde(data_points)
            #x_range = np.linspace(data_points.min(), data_points.max(), 200)
            #ax.plot(x_range, kde(x_range), color=color, lw=2, label=label if j==0 else "")

        ax.set_xlabel(x_lab)
        if j == 0:
            ax.set_ylabel(r"$\rho$")
            ax.legend(frameon=False, loc='upper right', fontsize='small')
        else:
            ax.tick_params(axis='y', labelleft=False)

        # Sublabel (e) and (f)
        ax.text(-0.05, 1.07, ["(e)", "(f)"][j], transform=ax.transAxes, fontweight='bold')




    if savefig:
        plt.savefig(savefig)
    plt.show()



log_path = "../logs/log_PROVA.csv"
vec_path = "../logs/eigen/"


#the states are vec0, vec1, bulk, vmax

# System 1: Low Gamma (Localized-ish)
bundle1 = get_aggregated_data_from_log(log_path, vec_path, target_state="vec0", target_beta=1.21, target_gamma=2.21)

# System 2: High Gamma (Extended/Bulk)
bundle2 = get_aggregated_data_from_log(log_path, vec_path, target_state="vec0", target_beta=10.01, target_gamma=8.01)

# Plot them side-by-side
plot_full_analysis(bundle1, bundle2, savefig="../figures/ground/description_vec0.pdf")


# System 1: Low Gamma (Localized-ish)
bundle1 = get_aggregated_data_from_log(log_path, vec_path, target_state="vmax", target_beta=1.21, target_gamma=2.21)

# System 2: High Gamma (Extended/Bulk)
bundle2 = get_aggregated_data_from_log(log_path, vec_path, target_state="vmax", target_beta=10.01, target_gamma=8.01)

# Plot them side-by-side
plot_full_analysis(bundle1, bundle2, savefig="../figures/ground/description_vecmax.pdf")



