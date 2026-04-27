import numpy as np
from plots import plot_all_idos, plot_all_idos_with_inset, plot_mean, plot_vec0_metrics_panel, create_comparison_figure
import utilities as ut
import time

def main():

    N = 2000
    J = 1.0
    k = 800
    noise=1e-10
    
    fillings = [1]
    T_min = 0.001
    T_max = 10000
    
    # --- deterministic graphs ---
    graphs = [
        #ut.ChainGraph(N, J, noise=noise, periodic=False),
        #ut.RangeChainGraph(N, J, R=2, noise=noise, periodic=False),
        #ut.RangeChainGraph(N, J, R=3, noise=noise, periodic=False),
        #ut.RangeChainGraph(N, J, R=4, noise=noise, periodic=False),
        #ut.SquareGraph(N, J, Lx=80, Ly=81, noise=noise, periodic=False),
        #ut.CubicGraph(N, J, Lx=18, Ly=19, Lz=17, noise=noise, periodic=False),
        #ut.HyperbolicGraph(J=J, p=7, q=3, noise=noise, nlayers=7),
        #ut.HyperbolicGraph(J=J, p=3, q=7, noise=noise, nlayers=8),
        #try this in the cluster: ut.HyperbolicGraph(J=J, p=3, q=10, nlayers=8),
    ]
    
    """
    while noise < 0.1:
        graphs += [ut.SquareGraph(N, J, Lx=80, Ly=81, noise=noise, periodic=False)]
        noise=noise*3
    """
    """
    Watts-Strogatz model
    
    ws_k = 4
    ws_p = 0
    
    while ws_p < 1:
        graphs += [ut.WattsStrogatz(N, J, ws_k=ws_k, ws_p=ws_p) for _ in range(num_samples)]
        ws_p+=0.05
        
    """



    """
    Configuration model. 
    
    for gamma in [2, 2.5, 3]:
        graphs +=[ut.configuration(N, J, gamma=gamma) for _ in range(num_samples)]
    """
    
    
    """
    Barabasi-Alber model
    for m in [2,5,10]:
        graphs +=[ut.barabasi(N, J, m=m) for _ in range(num_samples)]
    """
    
    #Real networks see what happens: geometric renormalization unravels self-fimilarity multiscale human connectome. navigable brain maps data zenodo. konect project network data/Icon networks database/Netzschleuder. Unipartide. Internet?
    
    dim=1
    beta=1.61
    gamma=2.01
    mean_degree=15
    num_samples=3
    nodes=[4000]
    
    for N in nodes:
        while gamma < 6.01:
            beta = 1.01
            while beta < 9.02:
                for i in range(num_samples):

                    g = ut.HyperbolicSD(
                        N, J,
                        dim=dim,
                        beta=beta,
                        gamma=gamma,
                        mean_degree=mean_degree,
                        output=f"../input/hyper_N_{N}_d_{dim}_b_{beta}_g_{gamma}_k_{mean_degree}_n_{noise}_{i}",
                        noise=noise,
                        it=i
                    )
                    #plot_enhanced_network(g)
                    
                    start_time = time.time()
                    # --- COMPUTE ---
                    eigenvalues, E0, vec0 = g.compute_eigenvalues(folder="../spectra/SD", force=True, save=False)

                    g.gap_ratio_unfolded(remove_edges=0.1, window_size=11)
                    g.compute_conden(fillings, T_min, T_max)

                    # --- SAVE ONLY WHAT YOU NEED ---
                    g.append_network_log(filename="log_ipr.csv", dist_filename="dist_ipr.csv")
                    
                    end_time = time.time()
                    duration = end_time - start_time

                    print(f"Iteration for N={g.N} took {duration:.2f} seconds")

                    # --- CLEAN MEMORY (important for large runs) ---
                    del g, eigenvalues, E0, vec0

                beta += 8
            print(f"\n\n Done for {gamma} value.\n\n")
            gamma += 0.5
    


    #This is a very unsuccesful structure, since i plot and compute, i should have this in different parts, save in the .log and later plot.
    
    #for g in graphs:
        #eigenvalues, E0, vec0 = g.compute_eigenvalues(folder="../spectra/SD",force=False, save=False)
        #print(f"Min = {eigenvalues.min()} and max = {eigenvalues.max()} eigenvalues of the spectra")
        #E_norm, IDOS = g.compute_idos(k=k)
        #idos_all[g.label] = (E_norm, IDOS)
        #print("Spectrum size different eigenvalues:", len(np.unique(eigenvalues)))
        
        #_,_,_ = g.gap_ratio_unfolded(remove_edges=0.1, window_size=11)
        #g.compute_conden(fillings, T_min, T_max)
        #g.plot_gap_ratio(remove_edges=0.20)
        #g.plot_ground_state(vec0, log_scale=False, cmap="plasma_r", use_networkx=True, show_edges=True)
        #g.plot_conden_frac()
        #g.append_network_log(filename = "log_sizes.csv")
    
    #plot_all_idos(idos_all, "../figures/idos_chain.pdf")
    #plot_all_idos_with_inset(idos_all,"../figures/idos/idos_inset_sd_neigh.pdf" )
    #plot_mean(graphs, "noise", "alpha", f"../figures/means/alpha_vs_noise_sq.pdf")
    #plot_mean(graphs, "noise", "Tc", f"../figures/means/Tc_vs_noise_sq.pdf")
    #plot_mean(graphs, "noise", "r_mean", f"../figures/means/r_mean_vs_noise_sq.pdf")
    #plot_vec0_metrics_panel(graphs, parameterx="beta", savefile=f"../figures/means/vec0_metrics_SD_g{gamma}.pdf")


    """
    Creates the basic lattices for our model. You define a specific graph, and in this case it creates 4 plots. Can be modified to the correct use.
    """
    #vec0=[0,0]
    #create_comparison_figure(g1, g2, g3, g4)

if __name__ == "__main__":
    main()
