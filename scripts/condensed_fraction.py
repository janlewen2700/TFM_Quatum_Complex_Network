import os
import glob
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from utilities import set_article_style

def bose(eigen, mu, T):
    eigen = np.asarray(eigen)
    x = (eigen - mu) / T

    n = np.empty_like(x)
    small = x < 1e-8
    large = x > 700  # np.exp(700) is near the float64 limit

    # Case 1: Small x (Expansion)
    n[small] = 1.0 / x[small]
    # Case 2: Large x (n approaches 0 or exp dominates)
    n[large] = 0.0
    # Case 3: Normal range
    mask = ~small & ~large
    n[mask] = 1.0 / (np.exp(x[mask]) - 1.0)

    return np.sum(n)

# ---------- Solve chemical potential ----------
def solve_mu(N, eigen, T):
    en0 = np.min(eigen)

    def f(mu):
        return bose(eigen, mu, T) - N

    # lower bound far below ground state
    mu_min = en0 - 100*T - 10.0
    mu_max = en0 - 1e-12  # must stay below ground state

    return brentq(f, mu_min, mu_max)


# ---------- Ground state occupation ----------
def ground_state_occupation(en0, mu, T):
    return float(1.0 / (np.exp((en0 - mu)/T) - 1.0))


# ---------- Maximum excited population ----------
def N_ex_max(eigen, T):
    eigen = np.asarray(eigen).ravel()
    en0 = np.min(eigen)
    T = float(T)

    excited = eigen[eigen > en0]
    x = (excited - en0) / T

    # small-x expansion for numerical stability
    small = x < 1e-8
    n = np.empty_like(x, dtype=float)
    n[small] = 1.0 / x[small]
    n[~small] = 1.0 / (np.exp(x[~small]) - 1.0)

    return float(np.sum(n))


# ---------- Find Tc from N = N_ex_max ----------
def find_Tc(N, eigen, T_min, T_max):
    """
    Finds Tc such that N = 2 * N_ex_max(Tc).
    This follows the paper's definition: Nc = 2 * N_ex_max.
    """
    def condition(T):
        # We assume mu = epsilon_0 to find the 'capacity'.
        return N_ex_max(eigen, T) - N

    # N_ex_max increases with T.
    # At low T, Nex < N (condition is negative).
    # At high T, Nex > N (condition is positive).
    return brentq(condition, T_min, T_max)
    
def fit_condensate_alpha(fill, Tc, T_array, fractions, fit_min_ratio=0.05, fit_max_ratio=0.95):

    import numpy as np
    from scipy.optimize import curve_fit

    T_array = np.asarray(T_array)
    fractions = np.asarray(fractions)

    mask = (
        (T_array > fit_min_ratio * Tc) &
        (T_array < fit_max_ratio * Tc) &
        (fractions > 0.0) &
        (fractions < 1.0)
    )

    def model_alpha(T, alpha):
        return 1.0 - (T / Tc) ** alpha

    alpha = np.nan
    alpha_err = np.nan

    if np.sum(mask) > 5:
        try:
            popt, pcov = curve_fit(
                model_alpha,
                T_array[mask],
                fractions[mask],
                p0=[1.5],
                bounds=(0.01, 10.0)
            )
            alpha = popt[0]
            alpha_err = np.sqrt(np.diag(pcov))[0]
        except RuntimeError:
            print(f"Alpha fit failed for filling {fill}")
    
    """
    if not hasattr(self, "Tc_dict"):
        self.Tc_dict = {}
    if not hasattr(self, "alpha_dict"):
        self.alpha_dict = {}
    if not hasattr(self, "alpha_err_dict"):
        self.alpha_err_dict = {}

    self.Tc_dict[fill] = Tc
    self.alpha_dict[fill] = alpha
    self.alpha_err_dict[fill] = alpha_err

    if np.isclose(fill, 1.0):
        self.Tc = Tc
        self.alpha = alpha
        self.alpha_err = alpha_err
    """
    return alpha, alpha_err
    
    

def extract_bec_data_from_file(file_path, fillings=[0.1, 0.5, 1.0]):
    """
    Reads a single .npz file and computes the condensation fractions 
    for the requested fillings.
    """
    # 1. Load the raw eigenvalues
    data = np.load(file_path)
    evals = data["evals"]
    evals = np.sort(np.real(evals)) # Ensure sorted real parts
    
    results_per_filling = {}

    # Define search range for Tc solver
    T_min_search, T_max_search = 1e-6, 500.0

    for fill in fillings:
        N = fill * len(evals)
        
        # 2. Solve for Tc (using the standalone find_Tc function)
        try:
            Tc = find_Tc(N, evals, T_min_search, T_max_search)
            
            # 3. Generate Temperature range and calculate n0/N
            T_array = np.linspace(1e-5 * Tc, 1.2 * Tc, 100)
            fractions = []
            
            for T in T_array:
                mu = solve_mu(N, evals, T)
                n0 = 1.0 / (np.exp((evals[0] - mu) / T) - 1.0)
                fractions.append(n0 / N)
            
            # 4. Fit alpha (if it's the main filling, e.g., 1.0)
            alpha, alpha_err = fit_condensate_alpha(fill, Tc, T_array, fractions, fit_min_ratio=0.05, fit_max_ratio=0.95)
            
            results_per_filling[fill] = {
                "Tc": Tc,
                "T_norm": T_array / Tc, # Normalized T for plotting
                "n0_norm": np.array(fractions),
                "alpha": alpha
            }
        except Exception as e:
            print(f"Failed to process filling {fill} in {file_path}: {e}")

    return results_per_filling



def plot_condensate_comparison(results_dict, savepath=None):
    set_article_style()
    fig, ax = plt.subplots() # REVTeX column width

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_dict)))

    for i, (fill, data) in enumerate(results_dict.items()):
        ax.scatter(data["T_norm"], data["n0_norm"],
                   label=f"f={fill}", s=4, color=colors[i], alpha=0.6)
        
        # If we have a fit for the filling, plot the dashed line
        if not np.isnan(data["alpha"]):
            #t_fine = np.linspace(0, 1, 100)
            #y_fit = 1.0 - t_fine**data["alpha"]
            #ax.plot(t_fine, y_fit, color=colors[i], linestyle='--', lw=1)
            
            if fill == 1.0:
                alpha_save = data["alpha"]

    # Add standard 3D reference line (alpha = 1.5)
    t_ref = np.linspace(0, 1, 100)
    ax.plot(t_ref, 1 - t_ref**1.5, 'k:', label="3D Theory", lw=1.2)

    ax.set_xlabel(r"$T / T_c$")
    ax.set_ylabel(r"$n_0 / N$")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    
    
    
def aggregate_fractions_across_files(files, fillings=[1.0]):
    """
    Computes mean and std error of n0/N across an ensemble.
    """
    # Grid of normalized temperature
    t_norm_grid = np.linspace(1e-5, 1.2, 100)
    
    ensemble_results = {fill: [] for fill in fillings}
    
    for f in files:
        data = np.load(f)
        evals = np.sort(np.real(data["evals"]))
        
        for fill in fillings:
            N = fill * len(evals)
            Tc = find_Tc(N, evals, 1e-6, 500.0) # Using the larger search range
            
            # Map the grid to actual temperatures for this Tc
            fractions = []
            for t_norm in t_norm_grid:
                T = t_norm * Tc
                mu = solve_mu(N, evals, T)
                n0 = 1.0 / (np.exp((evals[0] - mu)/T) - 1.0)
                fractions.append(n0 / N)
            
            ensemble_results[fill].append(fractions)
            
    # Calculate Mean and Std Error
    final_stats = {}
    for fill in fillings:
        arr = np.array(ensemble_results[fill]) # Shape: (realizations, 100)
        final_stats[fill] = {
            "t_norm": t_norm_grid,
            "mean": np.mean(arr, axis=0),
            "std_err": np.std(arr, axis=0) / np.sqrt(len(files))
        }
    return final_stats
    
def plot_three_regime_stack(ensemble_files_1, ensemble_files_2, lattice_file, fillings=[0.5, 1.0, 2.0], savepath=None):
    set_article_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.375, 5), sharex=True)
    fig.subplots_adjust(hspace=0.03)
    

    # data calculation:
    stats1 = aggregate_fractions_across_files(ensemble_files_1, fillings = fillings)
    stats2 = aggregate_fractions_across_files(ensemble_files_2, fillings = fillings)
    lat_res = extract_bec_data_from_file(lattice_file, fillings=fillings)
    
    for fill in fillings:
        ax1.errorbar(stats1[fill]["t_norm"], stats1[fill]["mean"], yerr=stats1[fill]["std_err"],
                 fmt='o', ms=2, alpha=0.8, label =fr"${fill}$")

        ax2.errorbar(stats2[fill]["t_norm"], stats2[fill]["mean"], yerr=stats2[fill]["std_err"],
                 fmt='s', ms=2, alpha=0.8)

        ax3.scatter(lat_res[fill]["T_norm"], lat_res[fill]["n0_norm"], s=3, alpha=0.8)

    # Clean up axes
    labels = [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$", r"$\mathbf{(c)}$"]
    axes = [ax1, ax2, ax3]
    for i, ax in enumerate(axes):
        # Add the 3D Reference line to all for comparison
        t_ref = np.linspace(0, 1, 100)
        ax.plot(t_ref, 1 - t_ref**1.5, 'k:', lw=2, alpha=0.8, zorder=10)
        ax.text(0.90, 0.90, labels[i], transform=ax.transAxes, fontweight='bold', va='top')
        ax.set_ylim(-0.05, 1.1)
    
        
        if ax!= ax3:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=None, bottom=True)
        else:
            ax.set_ylabel(r"$n_0/N$")
    
    leg=ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.15)
    ax1.text(0.15, 1.08, r"$\mathbf{Filling:}$", transform=ax1.transAxes,
         fontsize=10, va='center', ha='right')

    ax3.set_xlabel(r"$T/T_c$")
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


folder = "../spectra/SD"
files = glob.glob(os.path.join(folder, "SD_N=4000_D=1_B=1.*.npz"))
if not files:
    print("No files found matching that prefix.")
    


folder = "../spectra/SD"
files1 = glob.glob(os.path.join(folder, "SD_N=4000_D=1_B=1.21_G=2.*.npz"))
files2 = glob.glob(os.path.join(folder, "SD_N=4000_D=1_B=10.01_G=8.*.npz"))
files3 = "../spectra/H_3_7.npz"
plot_three_regime_stack(files1, files2, files3, savepath="../figures/cond_frac/Hyper_SD_comp.pdf")
