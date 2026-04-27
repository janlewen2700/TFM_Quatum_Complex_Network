import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# -----
#   Plots IDOS vs normalised energy
# ----

def set_article_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,

        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.linewidth": 1.0,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    

def plot_idos_vs_normalised_energy(E_norm, IDOS):
    plt.figure()
    plt.plot(E_norm, IDOS, label="IDOS")
    plt.plot([0,1],[0,1],'--',label="uniform DOS")

    plt.xlabel("Normalised Energy")
    plt.ylabel("IDOS")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_all_idos(idos_dict, savefile):

    set_article_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- Plot curves ----
    for name, (E_norm, IDOS) in idos_dict.items():
        ax.plot(E_norm, IDOS, linewidth=2, label=name)

    # ---- Axes formatting ----
    ax.set_xlabel("Normalised Energy")
    ax.set_ylabel("IDOS")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Remove top/right spines (clean journal look)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(savefile, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
    


def plot_all_idos_with_inset(idos_dict, savefile):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    import numpy as np
    set_article_style()
    # Increase the right margin of the figure to make room for the external inset
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Plot main curves ----
    for name, (E_norm, IDOS) in idos_dict.items():
        ax.plot(E_norm, IDOS, linewidth=2, label=name)

    # ---- Create the External Inset ----
    # bbox_to_anchor=(x, y, width, height) in axis coordinates
    # (1.05, 0.5) puts it to the right of the main plot
    ax_ins = inset_axes(ax, width="35%", height="40%", loc="center left",
                        bbox_to_anchor=(1.1, 0.3, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    
    for name, (E_norm, IDOS) in idos_dict.items():
        ax_ins.plot(E_norm, IDOS, linewidth=1.5)

    # ---- Inset Formatting ----
    x_limit = 0.08  # Updated to 0.08 as requested
    # We find the max IDOS value within this range to scale Y appropriately
    y_limit = 0
    for _, (E_norm, IDOS) in idos_dict.items():
        mask = E_norm <= x_limit
        if np.any(mask):
            y_limit = max(y_limit, np.max(IDOS[mask]))
    
    if y_limit < 0.01:
        y_limit = 0.01
        
    ax_ins.set_xlim(0, x_limit)
    ax_ins.set_ylim(0, y_limit * 1.1) # 10% headroom
    
    ax_ins.set_title(f"Low Energy Zoom (x < {x_limit})", fontsize=10)
    ax_ins.grid(True, linestyle=':', alpha=0.6)

    # Draw connector lines from the main plot to the external inset
    mark_inset(ax, ax_ins, loc1=2, loc2=3, fc="none", ec="0.5", linestyle="--", alpha=0.3)

    # ---- Main Axes formatting ----
    ax.set_xlabel("Normalised Energy")
    ax.set_ylabel("IDOS")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    # Use bbox_inches="tight" to ensure the external inset isn't cut off
    fig.savefig(savefile, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


#plots the gap ration for the full spectrum of the hamiltonian

def plot_gap_rat(self, bins=100, remove_edges=0, density=True, s_max=None):
    import numpy as np
    import matplotlib.pyplot as plt

    set_article_style()

    # r, r_mean, gaps = self.gap_ratio(remove_edges=remove_edges)
    r, r_mean, gaps = self.gap_ratio_unfolded(remove_edges=remove_edges)

    label = choose_spacing_family_from_r(self,r_mean)

    fig, ax = plt.subplots(figsize=(8, 6))

    # histogram of obtained spacings
    hist_vals, hist_edges, _ = ax.hist(
        gaps,
        bins=bins,
        density=density,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.6,
        label="Numerical data"
    )

    # choose x-range for theoretical curve
    if s_max is None:
        s_max = min(np.max(gaps), 3.0) if len(gaps) > 0 else 3.0

    s = np.linspace(0.0, s_max, 400)
    pdf = expected_spacing_pdf(self,s, label)

    ax.plot(
        s,
        pdf,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label=f"Expected {label}"
    )

    ax.set_xlabel("s")
    ax.set_ylabel("P(s)")
    ax.set_title(f"{self.label}   ⟨r⟩ = {r_mean:.4f}")

    ax.legend()
    fig.tight_layout()
    #fig.savefig(f"../figures/gap/gap_{self.label}_{self.noise}.pdf", bbox_inches="tight")
    #plt.show()
    plt.close(fig)

  
  
#these functions serve tto plot the theoretical curve

def choose_spacing_family_from_r(self, r_mean):
    r_poisson = 0.386
    r_goe = 0.536

    d_poisson = abs(r_mean - r_poisson)
    d_goe = abs(r_mean - r_goe)

    label = "Poisson" if d_poisson < d_goe else "Wigner-Dyson"

    self.spacing_fit_label = label
    return label
    
def expected_spacing_pdf(self, s, label):
    import numpy as np

    s = np.asarray(s)

    if label == "Poisson":
        return np.exp(-s)

    elif label == "Wigner-Dyson":
        return 0.5 * np.pi * s * np.exp(-np.pi * s**2 / 4.0)

    else:
        raise ValueError(f"Unknown spacing label: {label}")
        
        
  
  
"""
This function has to be added to the main pipeline, since it is the plotting for the given graph, i would like to reuse it later.
"""
        
def plot_n0_Tc_from_data(self):
    import matplotlib.pyplot as plt
    from plots import set_article_style

    if not hasattr(self, "n0_Tc_results"):
        raise ValueError("Run compute_n0_Tc() first")

    set_article_style()

    for fill, data in self.n0_Tc_results.items():
        Tc = data["Tc"]
        T_array = data["T_array"]
        fractions = data["fractions"]

        plt.scatter(T_array / Tc, fractions, label=f"Filling {fill}", s=5)

    # analytical lines
    t_norm = np.linspace(0, 1, 100)
    plt.plot(t_norm, 1 - t_norm**1.5, 'k--', label="Analytical (3D)")

    if hasattr(self, "alpha"):
        plt.plot(t_norm, 1 - t_norm**self.alpha, 'r--', label="fitted law")

    plt.xlabel(r"$T / T_c$")
    plt.ylabel(r"$n_0 / N$")
    plt.legend()
    plt.show()
    


#plots the ground state for the full spectrum of the hamiltonian
def plot_gro_sta(self, vec0, noise=0, log_scale=False, cmap="inferno", s=80, use_networkx=False, show_edges=False, ax=None):

    set_article_style()

    import numpy as np
    import matplotlib.pyplot as plt

    if not hasattr(self, "positions") or self.positions is None:
        print(f"Error: Lattice coordinates not available for {self.label}. Skipping plot.")
        return

    prob = np.abs(vec0)**2
    if log_scale:
        prob = np.log10(prob + 1e-16)

    x, y = self.positions[:, 0], self.positions[:, 1]
    

    if ax is None:
        fig, ax = plt.figure(figsize=(8, 6))

    if not use_networkx:
            # Original scatter plot
        #plt.scatter(x, y, c=prob, s=s, cmap=cmap, vmin=0, vmax=prob.max())
        nodes = ax.scatter(x, y, s=s)
        #plt.colorbar(label="|ψ₀|²" if not log_scale else "log10(|ψ₀|²)")
    else:
        import networkx as nx

        G = nx.Graph()

        # Add nodes
        for i, (xi, yi) in enumerate(self.positions):
            G.add_node(i, pos=(xi, yi))
            #G.add_node(i, pos=(xi, yi), weight=prob[i])

        # Add edges if available
        if hasattr(self, "edges") and show_edges:
            for edge in self.edges:
                if len(edge) == 2:
                    i, j = edge
                    G.add_edge(i, j)
                elif len(edge) == 3:
                    i, j, w = edge
                    G.add_edge(i, j, weight=w)

        pos_dict = nx.get_node_attributes(G, "pos")
        degrees = dict(G.degree())
        degree_values = np.array([degrees[i] for i in range(self.N)])
        sizes = 5 * (1 + (degree_values / degree_values.mean()+ 1e-6))

        #nodes = nx.draw_networkx_nodes(G, pos_dict, node_color=prob, node_size=s, cmap=cmap, vmin=0, vmax=prob.max(),)
        nodes = nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_size=sizes)

    if show_edges and hasattr(self, "edges"):
        nx.draw_networkx_edges(G, pos_dict, ax=ax, alpha=0.3)
    
                         
    ax.axis("off")
    #ax.title(f"Ground state distribution: {self.label}")
    
    if np.allclose(y, y[0]):
        plt.ylim(-100, 100)
        
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    
    if ax is None:
        plt.colorbar(nodes, label="|ψ₀|²" if not log_scale else "log10(|ψ₀|²)")
        plt.show()
        #plt.savefig(f"../figures/ground/ground_state_{self.label}_{noise}.pdf")
        plt.close()
 
    return nodes
 
def plot_vec0_metrics_panel(
    graphs,
    parameterx,
    metrics=None,
    savefile=None,
    use_sem=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    set_article_style()

    if metrics is None:
        metrics = [
            "participation_fraction",
            "avg_internal_degree",
            "n_components",
            "largest_component_fraction",
        ]

    if len(metrics) != 4:
        raise ValueError("This panel function expects exactly 4 metrics.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        grouped = defaultdict(list)

        for g in graphs:
            x = getattr(g, parameterx, None)
            vec0_metrics = getattr(g, "vec0_metrics", None)

            if x is None or vec0_metrics is None:
                continue
            if metric not in vec0_metrics:
                continue

            y = vec0_metrics[metric]

            if y is None:
                continue
            if isinstance(y, (float, int)) and np.isnan(y):
                continue

            grouped[x].append(float(y))

        xs = sorted(grouped.keys())
        means = []
        errors = []

        for x in xs:
            vals = np.array(grouped[x], dtype=float)
            mean = np.mean(vals)

            if len(vals) > 1:
                std = np.std(vals, ddof=1)
                err = std / np.sqrt(len(vals)) if use_sem else std
            else:
                err = 0.0

            means.append(mean)
            errors.append(err)

        ax.errorbar(
            xs,
            means,
            yerr=errors,
            fmt='o-',
            color='black',
            ecolor='black',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=5,
            capsize=3,
            linewidth=1.0
        )

        ax.set_xlabel(parameterx)
        ax.set_ylabel(metric)

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile)

    plt.show()
    
    
    
    
    
def plot_mean(graphs, parameterx, parametery, savefile=None, use_sem=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    set_article_style()
    plt.figure(figsize=(8, 6))

    grouped = defaultdict(list)

    # collect alpha values for each parameter value
    for g in graphs:
        x = getattr(g, parameterx, None)
        y = getattr(g, parametery, None)

        if x is None or y is None:
            continue
        if np.isnan(y):
            continue

        grouped[x].append(y)

    # sort by parameter value
    xs = sorted(grouped.keys())
    means = []
    errors = []

    for x in xs:
        vals = np.array(grouped[x], dtype=float)
        mean = np.mean(vals)

        if len(vals) > 1:
            std = np.std(vals, ddof=1)
            err = std / np.sqrt(len(vals)) if use_sem else std
        else:
            err = 0.0

        means.append(mean)
        errors.append(err)

    plt.errorbar(
        xs,
        means,
        yerr=errors,
        fmt='o',
        color='black',          # line + marker edge
        ecolor='black',         # error bar color
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=6,
        capsize=4,
        linewidth=1.2
    )

    plt.xlabel(parameterx)
    plt.ylabel(parametery)
    plt.xscale("log")
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
    
    
def create_comparison_figure(g1, g2, g3, g4):
    # 1. Create a figure with different projections
    vec0=[0]
    fig = plt.figure(figsize=(20, 5))
    
    
    # Subplot 1: Square (Standard Cartesian)
    ax1 = fig.add_subplot(1, 4, 1)
    # Subplot 2: Hyperbolic
    ax2 = fig.add_subplot(1, 4, 2)
    # Subplot 3: HyperbolicSD
    ax3 = fig.add_subplot(1, 4, 3, projection='polar')
    ax4 = fig.add_subplot(1, 4, 4, projection='polar')


    axes = [ax1, ax2, ax3, ax4]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    # 2. Call the functions and tell them WHERE to plot
    sc1 = g1.plot_ground_state(vec0, log_scale=False, use_networkx=True, show_edges=True, ax=ax1)
    
    sc2 = g2.plot_ground_state(vec0, log_scale=False, use_networkx=True, show_edges=True, ax=ax2)
    
    sc3 = g3.plot_ground_state(vec0, log_scale=False, use_networkx=True, show_edges=False, ax=ax3)
    
    sc4 = g4.plot_ground_state(vec0, log_scale=False, use_networkx=True, show_edges=False, ax=ax4)

    # 4. Standardize Geometry and Labels
    for i, ax in enumerate(axes):
        # Force the aspect ratio so nodes aren't stretched into ovals
        ax.set_aspect('equal', adjustable='box')
        
        # Consistent labeling using 'axes fraction'
        # (-0.05, 1.05) means 5% left of the plot and 5% above it
        ax.annotate(labels[i], xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=20, fontweight='bold', va='bottom', ha='right')

    # 5. Fixed Spacing Logic
    # Remove tight_layout as it overrides manual spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.05)
    plt.savefig("../figures/merged_networks_intro.pdf")
    plt.show()
    

