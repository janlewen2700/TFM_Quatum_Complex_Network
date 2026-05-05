import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plots import set_article_style

def plot_heatmap_from_log(
    logfile,
    value_col="r_mean",
    name="SD",
    N=2000,
    rel_tol=0.10,
    x_col="beta",
    y_col="gamma",
    agg="mean",
    savefile=None
):

    set_article_style()

    df = pd.read_csv(logfile)
    df = df[(df["name"] == name)]
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce").round(2)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").round(2)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = add_N_group_column(df, N_groups=N, rel_tol=rel_tol)
    df = df.dropna(subset=["N_group"])
    
    df = df[[x_col, y_col, value_col]].dropna()


    if df.empty:
        raise ValueError("No rows left after filtering.")

    grouped = df.groupby([y_col, x_col])[value_col].agg(agg).reset_index()
    pivot = grouped.pivot(index=y_col, columns=x_col, values=value_col)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    xvals = pivot.columns.to_numpy(dtype=float)
    yvals = pivot.index.to_numpy(dtype=float)
    zvals = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    mesh = ax.pcolormesh(xvals, yvals, zvals, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\mathbf{\langle \bar r \rangle}$")

    ax.set_xlabel(r"$\mathbf{\beta}$")
    ax.set_ylabel(r"$\mathbf{\gamma}$")
    #ax.set_title(f"{agg} {value_col} for {name}, N={N}")

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, bbox_inches="tight")

    plt.show()





def plot_mean_from_log(
    logfile,
    parameterx,
    parametery,
    parameter_control,
    control_val,
    N_groups,
    rel_tol=0.10,
    ave_k=None,
    name=None,
    savefile=None,
    use_sem=True,
    xscale=None,
    yscale=None,
):
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    from plots import set_article_style

    set_article_style()
    plt.figure(figsize=(8, 6))

    df = pd.read_csv(logfile)

    if name is not None:
        df = df[df["name"] == name]

    needed_cols = ["N", parameterx, parametery]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in log file.")

    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df[parameterx] = pd.to_numeric(df[parameterx], errors="coerce")
    df[parametery] = pd.to_numeric(df[parametery], errors="coerce")
    df[parameter_control] = pd.to_numeric(df[parameter_control], errors="coerce").round(2)
    df = df[df[parameter_control]==control_val]
    
    df = df.dropna(subset=needed_cols)
    
    df = add_N_group_column(df, N_groups=N_groups, rel_tol=rel_tol)
    df = df.dropna(subset=["N_group"])

    if df.empty:
        raise ValueError("No rows left after assigning N groups.")

    N_group_vals = sorted(df["N_group"].unique())
    
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    
    extrap_N = []
    extrap_beta = []
    extrap_beta_err =[]
    
    #modify enumerate depending on plot
    for i, Ng in enumerate(N_group_vals):
        sub = df[df["N_group"] == Ng]

        grouped = sub.groupby(parameterx)[parametery]

        xs = []
        means = []
        errors = []

        for x, vals in grouped:
            vals = np.array(vals, dtype=float)
            vals = vals[~np.isnan(vals)]

            if len(vals) == 0:
                continue

            mean = np.mean(vals)

            if len(vals) > 1:
                std = np.std(vals, ddof=1)
                err = std / np.sqrt(len(vals)) if use_sem else std
            else:
                err = 0.0

            xs.append(x)
            means.append(mean)
            errors.append(err)

        if len(xs) == 0:
            continue

        order = np.argsort(xs)
        xs = np.array(xs)[order]
        means = np.array(means)[order]
        errors = np.array(errors)[order]
        
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        #modify label depending on plot.
        plt.errorbar(
            xs,
            means,
            yerr=errors,
            label=fr"$N \approx {int(Ng)}$",
            #label=fr"$\langle k \rangle = {int(k)}$",
            linestyle=":",
            marker=marker,
            linewidth=1.2,
            markersize=4.5,
            capsize=2.5,
            elinewidth=0.9,
            markerfacecolor='white',
            markeredgewidth=1.0,
        )
        
        #The next block is added considering the size scaling.
        if Ng <2000:
            continue
            
        if len(means)>3:
            y_mid = (np.max(means)+np.min(means))/2
            
            n_samples = 100
            beta_samples = []
            for _ in range(n_samples):
                sampled_means = means + np.random.normal(0, errors)

                try:
                    f = interp1d(sampled_means, xs, bounds_error=False)
                    beta_samples.append(f(y_mid))
                except:
                    continue

            beta_mid = np.mean(beta_samples)
            beta_err = np.std(beta_samples)
                
            if not np.isnan(beta_mid):
                extrap_N.append(Ng)
                extrap_beta.append(beta_mid)
                extrap_beta_err.append(beta_err)

  
    if len(extrap_N) >= 2:
        def model(x, intercept, slope):
            return intercept + slope * x
        
        best_alpha = 0
        max_r2 = -np.inf
        best_intercept = 0
        best_slope=0
        
        for alpha in np.linspace(0.1, 2, 200):
            inv_N_alpha = 1.0 / (extrap_N**alpha)

            try:
                popt, pcov = curve_fit(model, inv_N_alpha, extrap_beta, sigma=extrap_beta_err, absolute_sigma=True)

                preds = model(inv_N_alpha, *popt)
                # weighted R²
                weights = 1 / np.array(extrap_beta_err)**2
                residuals = extrap_beta - preds

                ss_res = np.sum(weights * residuals**2)
                ss_tot = np.sum(weights * (extrap_beta - np.average(extrap_beta, weights=weights))**2)
                r2 = 1 - ss_res / ss_tot
                if r2 > max_r2:
                    max_r2 = r2
                    best_alpha = alpha
                    best_intercept = popt[0]
                    best_slope = popt[1]
                    best_intercept_err = np.sqrt(pcov[0, 0])

            except RuntimeError:
                continue

        print(f"Best Alpha: {best_alpha:.2f}")
        print(f"Best R^2: {max_r2:.5f}")
        
        print("\n--- Thermodynamic Limit Inference ---")
        for n, b in zip(extrap_N, extrap_beta):
            print(f"N = {int(n)}: Midpoint {parameterx} ≈ {b:.5f}")
        
        print(fr"Inferred {parameterx}_c (N -> ∞): {best_intercept:.5f} ± {best_intercept_err:.5f}")
        print(f"Scaling Slope: {best_slope:.5f}")
        
        # Optional: Add the inferred point to the plot or a new one
        plt.axvline(best_intercept, color='k', linestyle='--', label=f'Limit: {best_intercept:.3f}')
    else:
        print("\nNot enough valid data points for extrapolation (need at least 2 sizes > 500).")
        
        
        
    if parameterx == "beta":
        plt.xlabel(r"$\mathbf{\beta}$")
    elif parameterx == "gamma":
        plt.xlabel(r"$\mathbf{\gamma}$")
        
    if parametery == "r_mean":
        plt.ylabel(r"$\mathbf{\langle \bar r \rangle}$")
    elif parametery == "Tc":
        plt.ylabel(r"$\mathbf{Tc}$")

    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)

    plt.legend(title="Size group")
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, bbox_inches="tight")

    plt.show()
    plt.close()
    
    """
    inv_N = 1/(extrap_N**best_alpha)
    
    x_th = np.linspace(np.min(inv_N), np.max(inv_N), 200)
    y_th = best_slope*x_th + best_intercept
    
    plt.errorbar(inv_N, extrap_beta, yerr=extrap_beta_err, ecolor="k", capsize=4.0, linestyle="", marker="s", markeredgecolor="k", markerfacecolor='white', markeredgewidth=1.0)
    plt.plot(x_th, y_th, c="r",ls="--")
    plt.show()
    
    """


def add_N_group_column(df, N_groups, rel_tol=0.10, col_name="N_group"):
    df = df.copy()

    def assign(N):
        candidates = [Ng for Ng in N_groups if abs(N - Ng) / Ng <= rel_tol]
        if not candidates:
            return None
        return min(candidates, key=lambda Ng: abs(N - Ng))

    df[col_name] = df["N"].apply(assign)
    return df






# ground state histogram
def histogram_hubs(beta_val=2.01, gamma_val=2.01, file_dist="../logs/dist_log.csv", file_logs="../logs/log.csv", N=4000, savefile=None):

    df_dist = pd.read_csv(file_dist)
    df_main = pd.read_csv(file_logs)

    # Filter runs based on physics parameters
    df_main["beta"] = pd.to_numeric(df_main["beta"], errors="coerce").round(2)
    df_main["gamma"] = pd.to_numeric(df_main["gamma"], errors="coerce").round(2)
    
    target_runs = df_main[(df_main['beta'] == round(beta_val, 2)) &
                          (df_main['gamma'] == round(gamma_val, 2))]['run_id']

    filtered_dist = df_dist[df_dist['run_id'].isin(target_runs)].copy()
    
    if filtered_dist.empty:
        print(f"No runs found for beta={beta_val}, gamma={gamma_val}")
        return

    # 1. Logarithmic Binning
    max_k = filtered_dist['degree'].max()
    bins = np.logspace(np.log10(1), np.log10(max_k + 1), num=30)
    filtered_dist['bin_val'] = pd.cut(filtered_dist['degree'], bins=bins, labels=bins[:-1]).astype(float)

    # 2. Manual Intensity Calculation
    # First, aggregate the sums and counts PER RUN per bin
    run_binned = filtered_dist.groupby(['run_id', 'bin_val'], observed=True).agg(
        bin_sum_prob=('sum_prob', 'sum'),
        bin_deg_count=('deg_count', 'sum')
    ).reset_index()

    # Calculate Intensity for each bin in each run
    run_binned['intensity'] = run_binned['bin_sum_prob'] / run_binned['bin_deg_count']

    # 3. Final statistics across all realizations
    final_hist = run_binned.groupby('bin_val', observed=True)['intensity'].agg(['mean', 'std']).fillna(0)
    
    # Note: We scale P(k) just to see the slope comparison
    k_range = np.logspace(0, np.log10(max_k), 100)
    pk_theory = theoretical_p_k(k_range, gamma_val, k_avg=15) # From previous code

    # 4. Plotting
    set_article_style()
    plt.figure(figsize=(8, 6))

    # Actual Data (Intensity)
    plt.errorbar(final_hist.index, final_hist['mean'], yerr=final_hist['std'],
                 fmt='o', markersize=6, capsize=3, color='teal',
                 alpha=0.8, label=fr'Intensity $\langle |\phi|^2 \rangle$')

    # Optional: Theoretical Degree Distribution P(k) overlay
    plt.plot(k_range, pk_theory, color='darkorange', linestyle='--', alpha=0.6, label='Structural $P(k)$')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=1e-7)
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.xlabel(r'Degree ($k$)')
    plt.ylabel(r'Prob Density / Intensity')
    plt.title(fr'Localization Analysis ($\beta={beta_val}$, $\gamma={gamma_val}$)')
    plt.legend()
    plt.tight_layout()
    
    if savefile:
        # Get the directory path from the filename
        directory = os.path.dirname(savefile)
        
        # Create the directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        plt.savefig(savefile, bbox_inches="tight")
    plt.show()


def plot_degree_distribution(beta_val=1.0, gamma_val=1.0, k_avg=15, file_dist="../logs/dist_log.csv", file_logs="../logs/log.csv", N=4000, savefile=None):

    df_dist = pd.read_csv(file_dist)
    df_main = pd.read_csv(file_logs)

    # Filter runs based on physics parameters
    df_main["beta"] = pd.to_numeric(df_main["beta"], errors="coerce").round(2)
    df_main["gamma"] = pd.to_numeric(df_main["gamma"], errors="coerce").round(2)
    
    target_runs = df_main[(df_main['beta'] == round(beta_val, 2)) &
                          (df_main['gamma'] == round(gamma_val, 2))]['run_id']

    filtered_dist = df_dist[df_dist['run_id'].isin(target_runs)].copy()
    
    if filtered_dist.empty:
        print(f"No runs found for beta={beta_val}, gamma={gamma_val}")
        return
    
    # 1. Aggregate Observed Degrees across ALL runs in the file
    # We sum the counts for each degree across all realizations
    observed = filtered_dist.groupby('degree')['deg_count'].sum().reset_index()
    
    # Normalize to get P(k): Total counts / (Nodes * Number of Runs)
    num_runs = filtered_dist['run_id'].nunique()
    observed['p_k'] = observed['deg_count'] / (N * num_runs)

    # 2. Generate Theoretical Curve
    k_min = observed['degree'].min()
    max_k = observed['degree'].max()
    k_range = np.logspace(0, np.log10(max_k), 100)

    pk_theory = theoretical_p_k(k_range, gamma_val, k_avg)

    # 3. Plotting
    set_article_style()
    plt.figure(figsize=(8, 6))

    # Observed Data (Points)
    plt.scatter(observed['degree'], observed['p_k'],
                color='teal', alpha=0.6, s=20, label='Observed (Simulated)')

    # Theoretical Curve (Line)
    plt.plot(k_range, pk_theory, color='darkorange', linestyle='--',
             linewidth=2, label=fr'Theory $P(k)$ ($\gamma={gamma_val}$)')

    # Formatting for Power-Laws
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Degree ($k$)')
    plt.ylabel(r'Probability $P(k)$')
    plt.title('Degree Distribution: Structural Validation')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)
    
    # Set axis limits to focus on the data
    plt.xlim(left=1)
    plt.ylim(bottom=1e-5) # Adjust based on your N
    
    plt.tight_layout()
    
    if savefile:
        # Get the directory path from the filename
        directory = os.path.dirname(savefile)
        
        # Create the directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        plt.savefig(savefile, bbox_inches="tight")
    plt.show()





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






#plot_degree_distribution(beta_val = 1.61, gamma_val=2.41, k_avg=15, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", N=4000, savefile = f"../figures/vec_0/deg_b_1.61_g_2.41.pdf")

#plot_degree_distribution(beta_val = 1.61, gamma_val=6.01, k_avg=15, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", N=4000, savefile = f"../figures/vec_0/deg_b_1.61_g_6.01.pdf")

#plot_degree_distribution(beta_val = 9.61, gamma_val=6.01, k_avg=15, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", N=4000, savefile = f"../figures/vec_0/deg_b_9.61_g_6.01.pdf")

#plot_degree_distribution(beta_val = 9.61, gamma_val=8.01, k_avg=15, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", N=4000, savefile = f"../figures/vec_0/deg_b_9.61_g_8.01.pdf")

#plot_degree_distribution(beta_val = 1.61, gamma_val=4.01, k_avg=15, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", N=4000, savefile = f"../figures/vec_0/deg_b_1.61_g_4.01.pdf")

#histogram_hubs(beta_val=1.61, gamma_val=2.41, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", savefile = f"../figures/vec_0/vec_0_b_1.61_g_2.41.pdf")

#histogram_hubs(beta_val=1.61, gamma_val=4.01, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", savefile = f"../figures/vec_0/vec_0_b_1.61_g_4.01.pdf")

#histogram_hubs(beta_val=1.61, gamma_val=6.01, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", savefile = f"../figures/vec_0/vec_0_b_1.61_g_6.01.pdf")

#histogram_hubs(beta_val=9.61, gamma_val=6.01, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", savefile = f"../figures/vec_0/vec_0_b_9.61_g_6.01.pdf")

#histogram_hubs(beta_val=9.61, gamma_val=8.01, file_dist="../logs/dist_histogram_2.csv", file_logs="../logs/log_histogram_2.csv", savefile = f"../figures/vec_0/vec_0_b_9.61_g_8.01.pdf")

#plot_heatmap_from_log(logfile="../logs/log_sweep.csv", value_col="gap_0", name="SD", N=[4000], x_col="beta", y_col="gamma", savefile="../figures/heatmap/heatmap_gap_0_SD_N4000.pdf")
#plot_heatmap_from_log(logfile="../logs/log_sweep.csv", value_col="second_moment", name="SD", N=[4000], x_col="beta", y_col="gamma", savefile="../figures/heatmap/heatmap_second_moment_SD_N4000.pdf")




"""
plot_heatmap_from_log(
    logfile="../logs/log_new.csv",
    value_col="r_mean",
    name="SD",
    N=[4000],
    x_col="beta",
    y_col="gamma",
    savefile="../figures/heatmap/heatmap_rmean_SD_N4000.pdf"
)
"""

"""
plot_heatmap_from_log(
    logfile="../logs/log.csv",
    value_col="Tc",
    name="SD",
    N=[2000],
    x_col="beta",
    y_col="gamma",
    savefile="../figures/heatmap/heatmap_Tc_SD_N2000.pdf"
)

plot_mean_from_log(
    logfile="../logs/log_sizes.csv",
    parameterx="beta",
    parametery="alpha",
    parameter_control = "gamma",
    control_val = 6.01,
    N_groups=[500,1000,2000,4000,8000],
    rel_tol=0.10,
    name="SD",
    savefile="../figures/means/alpha_vs_beta_byNgroup.pdf"
)



plot_mean_from_log(
    logfile="../logs/log_sizes.csv",
    parameterx="beta",
    parametery="r_mean",
    parameter_control = "gamma",
    control_val = 6.01,
    N_groups=[500, 1000, 2000, 4000, 6000, 8000, 10000],
    rel_tol=0.10,
    name="SD",
    savefile="../figures/means/rmean_vs_beta_byNgroup.pdf"
)


plot_mean_from_log(
    logfile="../logs/log_k.csv",
    parameterx="beta",
    parametery="r_mean",
    parameter_control = "gamma",
    control_val = 6.01,
    N_groups=[2000],
    ave_k = [10,20,30,40],
    rel_tol=0.10,
    name="SD",
    savefile="../figures/means/rmean_vs_beta_byk.pdf"
)


plot_mean_from_log(
    logfile="../logs/log_ipr.csv",
    parameterx="gamma",
    parametery="r_mean",
    parameter_control = "beta",
    control_val = 1.01,
    N_groups=[4000],
    rel_tol=0.10,
    name="SD",
    #yscale="log",
    #savefile="../figures/means/ipr_vs_gamma_b1.01.pdf"
)
"""

plot_mean_from_log(
    logfile="../logs/log_sweep.csv",
    parameterx="gamma",
    parametery="gap_0",
    parameter_control = "beta",
    control_val = 10.01,
    N_groups=[4000],
    rel_tol=0.10,
    name="SD",
    yscale="log",
    #savefile="../figures/means/ipr_vs_gamma_b9.01.pdf"
)

