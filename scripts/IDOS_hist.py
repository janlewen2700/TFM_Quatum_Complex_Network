import os
import glob
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from utilities import set_article_style

def get_idos_data(files, energy_grid=None):
    """
    Computes IDOS (mean and std_err if ensemble, just y if single file).
    """
    if energy_grid is None:
        energy_grid = np.linspace(0, 1, 500)
    
    all_idos = []
    
    # If it's just one file (lattice), wrap it in a list
    if isinstance(files, str):
        files = [files]
        
    for f in files:
        data = np.load(f)
        # Normalize eigenvalues to [0, 1] for comparison
        evals = np.sort(np.real(data["evals"]))
        evals_norm = (evals - evals.min()) / (evals.max() - evals.min())
        
        # Calculate IDOS using searchsorted (very fast O(log N))
        # This counts how many eigenvalues are below each point in the grid
        idos = np.searchsorted(evals_norm, energy_grid) / len(evals_norm)
        all_idos.append(idos)
        
    all_idos = np.array(all_idos)
    return {
        "x": energy_grid,
        "mean": np.mean(all_idos, axis=0),
        "err": np.std(all_idos, axis=0) / np.sqrt(len(files)) if len(files) > 1 else None
    }
    

def plot_idos_stack(lattice_files, network_ensembles, labels_lat, labels_net, savefig=None):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import ScalarFormatter
    set_article_style()
    
    # 3.375 is standard column width; 7.5 height is good for a 2-panel stack
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.375, 5.0), sharex=True)
    fig.subplots_adjust(hspace=0.12)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x_grid = np.linspace(0, 1, 1000)
    x_ins_lim = 0.08

    # Loop through the two panels
    for i, ax in enumerate([ax1, ax2]):
        ax_ins = inset_axes(ax, width="35%", height="35%",
                    loc="upper left",
                    bbox_to_anchor=(0.1, 0.03, 0.9, 0.9), # (x, y, width, height)
                    bbox_transform=ax.transAxes)
        
        # Choose the correct data group: Lattices (ax1) or Networks (ax2)
        current_group = lattice_files if i == 0 else network_ensembles
        
        for j, files in enumerate(current_group):
        
            if not files or (isinstance(files, list) and len(files) == 0):
                print(f"Warning: No files found for {labels[j]} in panel {['a', 'b'][i]}")
                continue
            # get_idos_data handles both a single string or a list of strings
            res = get_idos_data(files, energy_grid=x_grid)
            
            # Use labels only for the top plot to avoid redundancy, or both
            current_label = labels_lat[j] if i == 0 else labels_net[j]
            
            # Plot Main
            ax.plot(res["x"], res["mean"], color=colors[j], label=current_label, lw=1.2)
            
            # Add error cloud ONLY if it's an ensemble (err exists)
            if res["err"] is not None:
                ax.fill_between(res["x"], res["mean"]-res["err"], res["mean"]+res["err"],
                                color=colors[j], alpha=0.8, zorder=1)
            
            # Plot Inset (Mirror the main plot data)
            ax_ins.plot(res["x"], res["mean"], color=colors[j], lw=1)
        
        if i!=0:
            ax_ins.set_ylim(0, 0.002)
            ax.set_ylabel(r"$\mathcal{I}(E)$")
        else:
            ax_ins.set_ylim(0, 0.05)
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=None, bottom=True)
                

        
        ax.set_ylim(0, 1.02)
        
        # Inset Formatting
        ax_ins.set_xlim(0, x_ins_lim)
        ax_ins.tick_params(axis='both', which='major', labelsize=7)
        ax_ins.grid(True, linestyle=':', alpha=0.4)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((0, 0))
        ax_ins.yaxis.set_major_formatter(yfmt)

    ax1.legend(loc="lower right", frameon=False)
    ax2.legend(loc="lower left", bbox_to_anchor=(0,0.1), frameon=False)
    ax2.set_xlabel(r"$(E-E_{min})/\Delta E$")
    
    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
    plt.show()



lattices = [
    "../spectra/H_7_3.npz",
    "../spectra/H_3_7.npz",
    "../spectra/C_20.npz",
    "../spectra/S_80.npz",
]

# 4 lists of files (one list per ensemble)
networks = [
    glob.glob("../spectra/SD/SD_N=4000_D=1_B=1.21_G=2.*.npz"),
    glob.glob("../spectra/SD/SD_N=4000_D=1_B=1.21_G=8.*.npz"),
    glob.glob("../spectra/SD/SD_N=4000_D=1_B=10.01_G=2.*.npz"),
    glob.glob("../spectra/SD/SD_N=4000_D=1_B=10.01_G=8.*.npz"),
]

labels_lat = [r"$\{7,3\}$", r"$\{3,7\}$", r"Cubic", r"Square"]
labels_net = [r"$\beta=1.21\; \gamma=2.21$",r"$\beta=1.21\; \gamma=8.01$",r"$\beta=10.01 \;\gamma=2.21$",r"$\beta=10.01\; \gamma=8.01$"]
plot_idos_stack(lattices, networks, labels_lat, labels_net, savefig="../figures/idos/idos_complete.pdf")
