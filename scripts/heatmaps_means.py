import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utilities import set_article_style

def plot_heatmap_from_log(
    logfile,
    value_col="r_mean",
    name="SD",
    N=2000,
    rel_tol=0.10,
    x_col="beta",
    y_col="gamma",
    agg="mean",
    ax=None,
    savefile=None,
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

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    mesh = ax.pcolormesh(
        xvals,
        yvals,
        zvals,
        cmap='viridis',
        shading='gouraud',
        edgecolors='none',
        rasterized=True  # Recommended for heatmaps in PDFs to keep file size small
    )
    
    pos = ax.get_position()
    # Place it above the plot
    cax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])
    
    # Get min/max for ticks
    z_min, z_max = np.nanmin(zvals), np.nanmax(zvals)
    
    cb = fig.colorbar(
        mesh,
        cax=cax,
        orientation='horizontal',
        ticks=[z_min, z_max] # Only show min and max
    )
    
    # Format to 1 or 2 decimals to save space
    cax.invert_xaxis()
    cb.ax.set_xticklabels([f"{z_min:.2f}", f"{z_max:.2f}"])
    
    
    # Move the label to the side of the colorbar or keep it on top
    cb.set_label(r"$\mathbf{\langle \bar r \rangle}$", labelpad=-25, x=0.5, fontweight='bold')
    cax.xaxis.set_ticks_position('top')
    ax.set_ylabel(r"$\mathbf{\gamma}$", rotation=0, y=0.48, labelpad=15)







def plot_mean_from_log(
    logfile,
    parameterx,
    parametery,
    parameter_control,
    control_val,
    N_groups,
    rel_tol=0.10,
    name=None,
    savefile=None,
    use_sem=True,
    xscale=None,
    yscale=None,
    ax=None,
):
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    set_article_style()
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

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
    
    cmap = plt.get_cmap('viridis')
    
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
        
        color_val = 0.2 + (i / len(N_group_vals))
        line_color = cmap(color_val)
        
        #modify label depending on plot.
        ax.errorbar(xs, means, yerr=errors, label=fr"$N \approx {int(Ng)}$", linestyle=":", color=line_color, marker=marker, linewidth=1, markersize=2, capsize=1, elinewidth=0.9, markerfacecolor='white', markeredgecolor=line_color, markeredgewidth=1.0,)
        
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

  
    if len(extrap_N) >= 2 and parametery == "r_mean":
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
        ax.axvline(best_intercept, color='k', linestyle='--', label=f'Limit: {best_intercept:.3f}')
    else:
        print("\nNot enough valid data points for extrapolation (need at least 2 sizes > 500).")
      
    if parametery == "r_mean":
        ax.set_ylim(0.38, 0.56)
        
        
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    
    if parametery!="gap0" and parametery!= "gap_0":
        pos = ax.get_position()
        if parameterx=="beta":
            cax = fig.add_axes([
                pos.x0 + 0.07 * pos.width,   # 5% from the left edge of the plot
                pos.y0 + 0.12 * pos.height,  # 5% from the bottom edge
                0.02,                        # Fixed width (2% of figure)
                0.10                         # Fixed height (15% of figure)
            ])
        elif parameterx=="gamma":
            cax = fig.add_axes([
                pos.x0 + 0.80 * pos.width,   # 5% from the left edge of the plot
                pos.y0 + 0.6 * pos.height,  # 5% from the bottom edge
                0.02,                        # Fixed width (2% of figure)
                0.10                         # Fixed height (15% of figure)
            ])

        # 2. Setup Normalization and Colorbar
        N_min, N_max = min(N_group_vals), max(N_group_vals)
        norm = mpl.colors.Normalize(vmin=N_min, vmax=N_max)
        
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation='vertical',
            ticks=[N_min, N_max]  # Only mark the boundaries
        )
        
        cb.solids.set_rasterized(False)

        # 3. Format boundary ticks (e.g., "500" and "10k" or "10^4")
        # Using 'left' alignment for ticks so they stay inside the axis if possible
        cax.set_yticklabels([f"{int(N_min)}", f"{int(N_max)}"], fontsize=7)
        cax.tick_params(axis='y')

        # 4. Thermodynamic Limit Label (placed directly ABOVE the colorbar)
        if len(extrap_N) >= 2 and parameterx=="beta":
            # We use text to simulate a legend entry
            # x=0.5 (centered), y=1.1 (just above the bar)
            cax.text(1.8, 1.2, r"$\mathbf{--\; \infty}$",
                        transform=cax.transAxes,
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        fontweight='bold')
        
        # 5. Optional: Vertical Label indicating the variable is N
        if parameterx=="beta":
            cax.set_ylabel(r"$\mathbf{N}$", fontsize=8,rotation=0, labelpad=-5, y=0.55)
        elif parameterx=="gamma":
            cax.set_ylabel(r"$\mathbf{N}$", fontsize=8,rotation=0, labelpad=-5, y=0.65)

        # 5. Hide labels as requested, but keep ticks
        ax.tick_params(axis='x', labelbottom=False, bottom=True)
        if parameterx == "beta":
            ax.tick_params(axis='y', labelleft=False, left=True)
            
    else:
        if parametery == "gap0" or parametery == "gap_0" :
            ax.set_yscale("log")
            ax.set_ylabel(r"$\Delta E_0$")
            ax.set_xlabel(r"$\gamma$")
            ax.tick_params(axis='both', left=True, right=False, bottom=True, top=False)
            
        

    
    
def plot_mean_by_k_fixed_N(
    logfile,
    parameterx,
    parametery,
    parameter_control,
    control_val,
    target_N,
    rel_tol=0.10,
    name=None,
    use_sem=True,
    xscale=None,
    yscale=None,
    ax=None,
):
    set_article_style()
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    df = pd.read_csv(logfile)

    if name is not None:
        df = df[df["name"] == name]

    # 1. Numeric conversion
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df[parameterx] = pd.to_numeric(df[parameterx], errors="coerce")
    df[parametery] = pd.to_numeric(df[parametery], errors="coerce")
    df[parameter_control] = pd.to_numeric(df[parameter_control], errors="coerce").round(2)
    
    # 2. Filter for N within tolerance of target_N
    mask_N = (df["N"] >= target_N * (1 - rel_tol)) & (df["N"] <= target_N * (1 + rel_tol))
    df = df[mask_N]

    # 3. Filter for the control parameter (e.g., gamma)
    df = df[df[parameter_control] == control_val]
    
    # 4. Clean up
    df = df.dropna(subset=[parameterx, parametery, "mean_degree"])

    if df.empty:
        raise ValueError(f"No rows found for N={target_N} (tol {rel_tol}) and {parameter_control}={control_val}")

    # 5. Group by ave_k
    # We round ave_k to integers or 1 decimal to avoid float grouping issues
    df["ave_k_rounded"] = df["mean_degree"].round(1)
    k_vals = sorted(df["ave_k_rounded"].unique())
    
    cmap = plt.get_cmap("viridis")
    
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']

    for i, k in enumerate(k_vals):
        sub = df[df["ave_k_rounded"] == k]
        
        # Group by x-axis (e.g., beta) to get means/errors
        grouped = sub.groupby(parameterx)[parametery]

        xs, means, errors = [], [], []

        for x_val, vals in grouped:
            vals = vals.to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0: continue

            xs.append(x_val)
            means.append(np.mean(vals))
            if len(vals) > 1:
                std = np.std(vals, ddof=1)
                errors.append(std / np.sqrt(len(vals)) if use_sem else std)
            else:
                errors.append(0.0)

        # Sort and plot
        order = np.argsort(xs)
        
        color_val = 0.2 + (i / len(k_vals))
        line_color = cmap(color_val)
        
        ax.errorbar(
            np.array(xs)[order],
            np.array(means)[order],
            yerr=np.array(errors)[order],
            label=fr"$\langle k \rangle = {k}$",
            linestyle=":",
            color=line_color,
            marker=markers[i % len(markers)],
            linewidth=1,
            markersize=3,
            capsize=1.5,
            markerfacecolor='white',
            markeredgecolor = line_color,
            markeredgewidth=1.0
        )


    pos = ax.get_position()
    
    if parameterx == "beta":
        cax = fig.add_axes([
            pos.x0 + 0.07 * pos.width,   # 5% from the left edge of the plot
            pos.y0 + 0.22 * pos.height,  # 5% from the bottom edge
            0.02,                        # Fixed width (2% of figure)
            0.10                         # Fixed height (15% of figure)
        ])
    elif parameterx=="gamma":
        cax = fig.add_axes([
            pos.x0 + 0.80 * pos.width,   # 5% from the left edge of the plot
            pos.y0 + 0.60 * pos.height,  # 5% from the bottom edge
            0.02,                        # Fixed width (2% of figure)
            0.10                         # Fixed height (15% of figure)
        ])

    # 2. Setup Normalization and Colorbar
    k_min, k_max = min(k_vals), max(k_vals)
    norm = mpl.colors.Normalize(vmin=k_min, vmax=k_max)
    
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation='vertical',
        ticks=[k_min, k_max]
    )
    cb.solids.set_rasterized(False)

    # 3. Format boundary ticks (e.g., "500" and "10k" or "10^4")
    # Using 'left' alignment for ticks so they stay inside the axis if possible
    cax.set_yticklabels([f"{int(k_min)}", f"{int(k_max)}"], fontsize=8)
    cax.tick_params(axis='y')
    
    # 5. Optional: Vertical Label indicating the variable is N
    if parameterx=="beta":
        cax.set_ylabel(r"$\mathbf{\langle k \rangle}$", fontsize=8, rotation=0, labelpad=0, y=0.6)
    elif parameterx=="gamma":
        cax.set_ylabel(r"$\mathbf{\langle k \rangle}$", fontsize=8, rotation=0, labelpad=0, y=0.65)

    # Labeling
    if parameterx == "beta":
        ax.set_xlabel(r"$\beta$")
    elif parameterx == "gamma":
        ax.set_xlabel(r"$\gamma$")
        
    if parametery == "r_mean":
        ax.set_ylabel(r"$\mathbf{\langle \bar r \rangle}$", labelpad=2, rotation=0, y=0.48)
        ax.set_ylim(0.38, 0.56)
    elif parametery == "Tc":
        ax.set_ylabel(r"$\mathbf{Tc}$", labelpad=5, rotation=0, y=0.48)



def add_N_group_column(df, N_groups, rel_tol=0.10, col_name="N_group"):
    df = df.copy()

    def assign(N):
        candidates = [Ng for Ng in N_groups if abs(N - Ng) / Ng <= rel_tol]
        if not candidates:
            return None
        return min(candidates, key=lambda Ng: abs(N - Ng))

    df[col_name] = df["N"].apply(assign)
    return df
    
    
    
def plot_combined_figure(logsweep="../logs/log.csv", logsize="../logs/log.csv", logk="../logs/log.csv", value_col="r_mean", savefig=None):
    # 1. Setup the figure for a two-column document

    set_article_style()
    
    if value_col == "r_mean":
        fig= plt.figure(figsize=(3.375, 6.5))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.2, wspace=0.2)

        # 2. Call heatmap (Subplot A)
        ax1 = fig.add_subplot(gs[0, :])
        plot_heatmap_from_log(logfile=logsweep, value_col=value_col, name="SD", N=[4000], x_col="beta", y_col="gamma", ax=ax1)
        ax1.text(-0.17, 1.27, "(a)", transform=ax1.transAxes, fontweight='bold')

        # 3. Call mean/scaling plot (Subplot B)
        ax2 = fig.add_subplot(gs[1, :])
        plot_mean_from_log(logfile=logsize, parameterx="beta", parametery=value_col, parameter_control = "gamma", control_val = 6.01, N_groups=[500, 1000, 2000, 4000, 6000, 8000], rel_tol=0.10, name="SD", ax=ax2)
        ax2.text(-0.17, 1.07, "(b)", transform=ax2.transAxes, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[2, :])
        plot_mean_by_k_fixed_N(logfile=logk, parameterx="beta", parametery=value_col, parameter_control = "gamma", control_val = 6.01, target_N=2000, rel_tol=0.10, name="SD", ax=ax3)
        
        pos = ax3.get_position()
        ax3.set_position([pos.x0, pos.y0+0.03, pos.width, pos.height])
        ax3.text(-0.17, 1.07, "(c)", transform=ax3.transAxes, fontweight='bold')
        
    elif value_col == "Tc":
        fig= plt.figure(figsize=(3.375, 3.5))
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.2, wspace=0.2)

        # 3. Call mean/scaling plot (Subplot B)
        ax1 = fig.add_subplot(gs[0, :])
        plot_mean_from_log(logfile=logsize, parameterx="gamma", parametery=value_col, parameter_control = "beta", control_val = 10.01, N_groups=[500, 1000, 2000, 4000, 6000, 8000, 10000], rel_tol=0.10, name="SD", ax=ax1)
        #ax1.text(-0.17, 1.07, "(a)", transform=ax1.transAxes, fontweight='bold')
        
        ax2 = fig.add_subplot(gs[1, :])
        plot_mean_by_k_fixed_N(logfile=logk, parameterx="gamma", parametery=value_col, parameter_control = "beta", control_val = 10.01, target_N=4000, rel_tol=0.10, name="SD", ax=ax2)
        
        pos = ax2.get_position()
        ax2.set_position([pos.x0, pos.y0+0.03, pos.width, pos.height])
        #ax2.text(-0.17, 1.07, "(b)", transform=ax2.transAxes, fontweight='bold')
        
    elif value_col == "gap0" or value_col == "gap_0":
        fig= plt.figure(figsize=(3.375, 2))
        gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.2, wspace=0.2)

        # 3. Call mean/scaling plot (Subplot B)
        ax1 = fig.add_subplot(gs[0, :])
        plot_mean_from_log(logfile=logsize, parameterx="gamma", parametery=value_col, parameter_control = "beta", control_val = 10.01, N_groups=[4000], rel_tol=0.10, name="SD", ax=ax1)
    
    
    
    
    if savefig:
        plt.savefig(savefig, dpi=600, bbox_inches='tight')
    
    plt.show()


#plot_combined_figure(logsweep="../logs/log_sweep.csv", logsize="../logs/log_sizes.csv", logk="../logs/log_k.csv", value_col="r_mean", savefig="../figures/heatmap/heat_r_size.pdf")

#plot_combined_figure(logsize="../logs/log_size_TC.csv", logk="../logs/log_k_TC.csv", value_col="Tc", savefig="../figures/heatmap/TC_size_k.pdf")

plot_combined_figure(logsize="../logs/log_size_TC.csv", value_col="gap0", savefig="../figures/heatmap/gap0_value.pdf")


