import numpy as np
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd

import graph_gen as gg
from hamiltonian import build_sparse_hamiltonian
from diagonal import diagonalise_and_analyse, compute_load_eigenvalues, compute_save_eigenvectors, compute_gap_ratio, compute_gap_ratio_unfolded, compute_ground_state_metrics
from plots import plot_gap_rat, plot_gro_sta, plot_n0_Tc_from_data
from plot_hypermap import plot_ground_hyper, plot_ground_hyper_simple
from condensate import compute_n0_Tc


class BaseGraph:
    def compute_idos(self, k=150):
        raise NotImplementedError



# -----
# We first create a class for the deterministic graphs, with each of the possible lattices
# -----
class DeterministicGraph(BaseGraph):

    def __init__(self, N, J, noise, periodic=False, label=None):
        self.N = N
        self.J = J
        self.noise=noise
        self.periodic = periodic
        self.label = label or self.__class__.__name__
        self._eigenvalues = None
        self._ground_energy = None
        self._ground_state = None
        self.edges, self.epsilon = self._generate_graph()

    def _generate_graph(self):
        raise NotImplementedError

    def build_hamiltonian(self):
        return build_sparse_hamiltonian(self.N, self.edges, self.J, self.epsilon)

    def compute_eigenvalues(self, folder="../spectra", force=False, save=True):
        return compute_load_eigenvalues(self, folder=folder, force=force, save=save)
        
    def compute_eigenvectors(self, folder="../spectra"):
        return compute_save_eigenvectors(self, folder=folder)


    def compute_idos(self, k=150):
        evals,_,_ = self.compute_eigenvalues()

        E_min, E_max = evals.min(), evals.max()
        E_norm = (evals - E_min) / (E_max - E_min)

        m = len(evals)
        IDOS = np.arange(1, m+1) / m

        return E_norm, IDOS
        
        
    #simple spacings
    def gap_ratio(self, remove_edges=0):
        
        compute_gap_ratio(self, remove_edges=remove_edges)
        
        
    #unfolded spacings
    def gap_ratio_unfolded(self, density_threshold=0.05, window_size=11):
    
        return compute_gap_ratio_unfolded(self, density_threshold=density_threshold, window_size=window_size)
    
    
    #plot spacings with mean gap ratio
    def plot_gap_ratio(self, bins=100, remove_edges=0.10, density=True):
        
        plot_gap_rat(self, bins=bins, remove_edges=remove_edges, density=density)
    
    
    
    
    
    def plot_ground_state(self, vec0, log_scale=False, cmap="inferno",
                      s=10, use_networkx=False, show_edges=True, ax=None):

        # Case 1: NOT SD → use normal lattice plot
        if not self.label.startswith("SD"):
            plot_gro_sta(
                self,
                vec0,
                noise=self.noise,
                log_scale=log_scale,
                cmap=cmap,
                s=s,
                use_networkx=use_networkx,
                show_edges=show_edges,
                ax=ax
            )

        # Case 2: SD → use hypermap plot
        else:
            file_edgelist = self.output + ".edge"
            file_coor     = self.output + ".gen_coord"
            
            """
            Tc = 0.5          # or wherever you store it
            Scale = s         # reuse marker size
            fname = os.path.join(self.output, f"ground_state_{self.label}")

            plot_ground_hyper(
                file_edgelist,
                file_coor,
                vec0,
                Tc,
                Scale,
                fname,
                log_scale=log_scale,
                cmap=cmap
            )
            """
            if self.dim==1:
                plot_ground_hyper_simple(self, vec0, file_edgelist, file_coor, log_scale=log_scale, cmap=cmap, show_edges=show_edges, ax=ax)
            else:
                print("DIMENSION to large for plotting.")

    
    #computes a fraction n0/N
    def compute_conden(self, fillings, T_min, T_max):
        
        return compute_n0_Tc(self, fillings, T_min, T_max)
       
    #plots the fraction
    def plot_conden_frac(self):
        
        plot_n0_Tc_from_data(self)
        
    def compute_vec0_metrics(self, vec0, threshold, use_prob=True):
       
        return compute_ground_state_metrics(self, vec0, threshold, use_prob=use_prob)
    
    
    
    
    
    def _spectrum_filename(self, folder="../spectra"):
        os.makedirs(folder, exist_ok=True)

        filename = (
            f"{self.label}"
            f"_N{self.N}"
            f"_J{self.J}"
            f"_n{self.noise}"
            f"_periodic{self.periodic}.npz"
        )

        return os.path.join(folder, filename)


    def append_network_log(self, folder="../logs", filename="log.csv", dist_filename="dist_log.csv", vec_folder="../logs/eigen"):
        import csv
        import os
        from datetime import datetime

        os.makedirs(folder, exist_ok=True)
        os.makedirs(vec_folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        dist_filepath = os.path.join(folder, dist_filename)
        
        # Generate a unique ID for this specific run to link the two files
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        metrics = getattr(self, "vec0_metrics", {})
        
        # --- Part A: Main Scalar Log ---
        fieldnames = [
            "run_id", "name", "N", "J", "noise", "periodic", "beta", "gamma",
            "mean_degree", "HypR", "first_moment", "second_moment", "r_mean", "dyson_beta", "alpha", "Tc", "gap0", "ipr", "dif_radius", "participation_fraction",
            "avg_internal_degree", "n_components", "largest_component_fraction"
        ]

        row = {
            "run_id": run_id,
            "name": self.name,
            "N": self.N,
            "J": self.J,
            "noise": self.noise,
            "periodic": self.periodic,
            "beta": getattr(self, "beta", None),
            "gamma": getattr(self, "gamma", None),
            "mean_degree": getattr(self, "mean_degree", None),
            "HypR": getattr(self, "HypR", None),
            "first_moment": getattr(self, "first_moment", None),
            "second_moment": getattr(self, "second_moment", None),
            "r_mean": getattr(self, "r_mean", None),
            "dyson_beta": getattr(self, "dyson_beta", None),
            "alpha": getattr(self, "alpha", None),
            "Tc": getattr(self, "Tc", None),
            "gap0": getattr(self, "gap0", None),
            "ipr": metrics.get("ipr"),
            "dif_radius": metrics.get("dif_radius"),
            "participation_fraction": metrics.get("participation_fraction"),
            "avg_internal_degree": metrics.get("avg_internal_degree"),
            "n_components": metrics.get("n_components"),
            "largest_component_fraction": metrics.get("largest_component_fraction"),
        }

        file_exists = os.path.isfile(filepath)
        with open(filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # --- Part B: Distribution Log (The new part) ---
        # This saves (Degree, Probability) pairs for this specific run
        deg_prob_map = metrics.get("deg_prob_map", {})
        deg_node_count = metrics.get("deg_node_count", {})
        if deg_prob_map:
            dist_file_exists = os.path.isfile(dist_filepath)
            with open(dist_filepath, mode="a", newline="") as f:
                dist_writer = csv.writer(f)
                if not dist_file_exists:
                    dist_writer.writerow(["run_id", "degree", "sum_prob", "deg_count"])
                
                for deg, p_sum in deg_prob_map.items():
                    k_count = deg_node_count.get(deg, 1)
                    dist_writer.writerow([run_id, deg, p_sum, k_count])
                
        # --- Part C: Eigenstate Binary Log ---
        # We save the raw vectors for vec0, vec1, bulk (middle), and max
        vec_filename = f"{run_id}.npz"
        vec_path = os.path.join(vec_folder, vec_filename)
        
        if hasattr(self, "eigenstates"):
            np.savez_compressed(
                vec_path,
                vec0=self.eigenstates.get("vec0"),
                vec1=self.eigenstates.get("vec1"),
                bulk=self.eigenstates.get("bulk"),
                vmax=self.eigenstates.get("max"),
                radius=self.coords['radius'].values,
                theta=self.coords['theta'].values,
                degrees=np.array(self.degrees),
            )



# --- Chain (same as range R=1) ---
class ChainGraph(DeterministicGraph):
    def __init__(self, N, J, noise, periodic=False):
        label = "Chain (PBC)" if periodic else "Chain (OBC)"
        super().__init__(N, J, noise, periodic=periodic, label=label)

    def _generate_graph(self):
        edges, self.positions = gg._chain_graph(self.N, J=self.J, periodic=self.periodic)
        print(f"noise is: {self.noise}")
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
        else:
            epsilon = np.zeros(self.N)
            
        return edges, epsilon


# --- Range Chain ---
class RangeChainGraph(DeterministicGraph):
    def __init__(self, N, J=1.0, R=2, noise=0, periodic=False):
        self.R = R
        label = f"Range chain R={R}"
        if periodic:
            label += " (PBC)"
        super().__init__(N, J, periodic=periodic, noise=noise, label=label)

    def _generate_graph(self, noise=0):
        edges, self.positions = gg._chain_range_graph(self.N, R=self.R, J=self.J, periodic=self.periodic)
        
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
        else:
            epsilon = np.zeros(self.N)
        return edges, epsilon


# --- Square Lattice ---
class SquareGraph(DeterministicGraph):
    def __init__(self, N, J=1.0, Lx=None, Ly=None, noise=0, periodic=False):
        self.Lx = Lx
        self.Ly = Ly

        # temporary dims just for label
        Lx_tmp = Lx or int(np.sqrt(N))
        Ly_tmp = Ly or (N // Lx_tmp)

        label = f"Square {Lx_tmp}x{Ly_tmp}"
        if periodic:
            label += " (PBC)"

        super().__init__(N, J, noise=noise, periodic=periodic, label=label)

    def _generate_graph(self):
        Lx = self.Lx or int(np.sqrt(self.N))
        Ly = self.Ly or (self.N // Lx)
        edges, self.positions = gg._square_graph(Lx, Ly, J=self.J, periodic=self.periodic)
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=Lx*Ly)
        else:
            epsilon = np.zeros(Lx*Ly)
        self.N = Lx*Ly
        return edges, epsilon


# --- Cubic Lattice ---
class CubicGraph(DeterministicGraph):
    def __init__(self, N, J=1.0, Lx=None, Ly=None, Lz=None, noise=0, periodic=False):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        L = int(round(N**(1/3)))
        Lx_tmp = Lx or L
        Ly_tmp = Ly or L
        Lz_tmp = Lz or L

        label = f"Cubic {Lx_tmp}x{Ly_tmp}x{Lz_tmp}"
        if periodic:
            label += " (PBC)"

        super().__init__(N, J, noise=noise, periodic=periodic, label=label)

    def _generate_graph(self):
        L = int(round(self.N**(1/3)))
        Lx = self.Lx or L
        Ly = self.Ly or L
        Lz = self.Lz or L
        edges = gg._cubic_graph(Lx, Ly, Lz, J=self.J, periodic=self.periodic)
        
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=Lx*Ly*Lz)
        else:
            epsilon = np.zeros(Lx*Ly*Lz)
        self.N = Lx*Ly*Lz
        return edges, epsilon


# --- Hyperbolic {p,q} Lattice ---
class HyperbolicGraph(DeterministicGraph):
    def __init__(self, J=1.0, p=7, q=3, nlayers=7, noise=0):
        self.p = p
        self.q = q
        self.nlayers = nlayers
        self.N = None  # store the target number of vertices
        self.J = J

        label = f"Hyperbolic {{{p},{q}}}"
        super().__init__(N=None, J=J, noise=noise, label=label)

    def _generate_graph(self):
            edges, N, self.positions = gg._tiling_graph(self.p, self.q, n_layers=self.nlayers, J=self.J)

            self.N = N  # <-- set correct size here
            if self.noise!=0:
                epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
            else:
                epsilon = np.zeros(self.N)

            return edges, epsilon






# -----
# We now create a class for the random graphs, with each of the possible generators
# -----
class RandomGraphEnsemble(BaseGraph):
    """
    Random graph ensemble with averaged IDOS.
    """

    def __init__(self, name, N, J=1.0, noise=0, num_samples=10, label=None, **kwargs):
        self.name = name
        self.N = N
        self.J = J
        self.num_samples = num_samples
        self.kwargs = kwargs
        self.label = label or name

    def _generate_single_graph(self, noise=0):
        raise NotImplementedError

    def compute_idos(self, k=150, noise=0):

        common_E = np.linspace(0,1,k)
        IDOS_avg = np.zeros_like(common_E)

        for _ in range(self.num_samples):
            edges, epsilon = self._generate_single_graph(noise=noise)
            H = build_sparse_hamiltonian(self.N, edges, epsilon)
            E_norm, IDOS = diagonalise_and_analyse(H, k=k)
            IDOS_avg += np.interp(common_E, E_norm, IDOS)

        IDOS_avg /= self.num_samples
        return common_E, IDOS_avg
        



class WattsStrogatz(DeterministicGraph):

    def __init__(self, N, J=1.0, ws_k=4, ws_p=0.1, noise=0):
        label = f"Watts–Strogatz k={ws_k}, p={ws_p}"
        self.ws_k=ws_k
        self.ws_p=ws_p
        super().__init__(N, J, noise=noise, label=label)

    def _generate_graph(self):
        G = nx.watts_strogatz_graph(self.N, self.ws_k, self.ws_p)
        
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
        G = nx.convert_node_labels_to_integers(G)
        # Update internal N to reflect LCC size
        N_real = G.number_of_nodes()
        #print(f"LCC Size: {N_real}/{self.N}")
        self.N=N_real

        degrees = [d for n, d in G.degree()]
        max_deg = np.max(degrees)
        avg_deg = np.mean(degrees)
        
        print(max_deg, avg_deg)
        #self.label += f" (k_max={max_deg}, k_avg={avg_deg:.2f})"

        pos_dict = nx.circular_layout(G)
        self.positions = np.array([pos_dict[i] for i in range(self.N)])
        
        
        edges = [(i, j, self.J) for i, j in G.edges()]
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
        else:
            epsilon = np.zeros(self.N)
            
            
        return edges, epsilon
        



class HyperbolicSD(DeterministicGraph):

    def __init__(self, N, J=1.0, dim=1, beta=3, gamma=1.5, mean_degree=10, output = "../input", noise=0, it=0):
        label = f"SD_N={N}_D={dim}_B={beta}_G={gamma}_K={mean_degree}_i{it}"
        self.name = "SD"
        self.dim=dim
        self.beta=beta
        self.gamma=gamma
        self.mean_degree=mean_degree
        self.output = output
        self.it=it #this will serve as the seed, it is just the number range
        super().__init__(N, J, noise=noise, label=label)

    
    def _generate_graph(self):
        import subprocess
        import os
        
        #We loop in order to get a network with the significant amount of nodes
        for _ in range(5):
            
            cmd = [
                "./genSD",
                "-d", str(self.dim),
                "-b", str(self.beta),
                "-n", str(self.N),
                "-g", str(self.gamma),
                "-k", str(self.mean_degree),
                "-o", str(self.output),
                "-s", str(np.random.randint(100000)),
                "-v",
            ]
            print("Running:", cmd)
            with open(os.devnull, "w") as fnull:
                subprocess.run(cmd, stdout=fnull, stderr=fnull, check=True)
            
            G = nx.read_edgelist(f"{self.output}.edge")
            
            lcc_nodes = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc_nodes).copy()
            

            N_real = G.number_of_nodes()
            print(f"LCC Size: {N_real}/{self.N}")
            
            if abs(N_real - self.N) / self.N <= 0.1:
                break
            
        old_N=self.N
        self.N=N_real
        
        #we relabel the nodes and add the coordinates for future working
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
        raw_coords = pd.read_csv(f"{self.output}.gen_coord", sep=r"\s+",
                                 names=['vertex', 'kappa', 'radius', 'theta', 'real_deg', 'exp_deg', 'tmp'],
                                 header=0)
        if 'tmp' in raw_coords.columns: raw_coords.drop('tmp', axis=1, inplace=True)
                                 
        with open(f"{self.output}.edge", 'r') as f:
            for line in f:
                if "mu:" in line:
                    mu = float(line.split(":")[-1].strip())
                    break

        id_map = nx.get_node_attributes(G, 'original_id')
        ordered_ids = [id_map[i] for i in range(N_real)]
        
        raw_coords['vertex'] = raw_coords['vertex'].astype(str)
        if not raw_coords['vertex'].iloc[0].startswith('v'):
            raw_coords['vertex'] = 'v' + raw_coords['vertex']
        
        self.coords = raw_coords.set_index('vertex').loc[ordered_ids].reset_index()
        
        node_degrees = np.array([G.degree(i) for i in range(self.N)])
        self.degrees = node_degrees
        max_deg = np.max(node_degrees)
        avg_deg = np.mean(node_degrees)
        second_moment = np.mean(np.square(node_degrees))
        variance = second_moment - avg_deg**2
        
        self.second_moment = second_moment
        self.first_moment = avg_deg
        
        print(max_deg, avg_deg)
        
        if self.gamma > 2:
            exponent = (2-self.gamma)/(self.gamma-1)
            k_0 = (1-1/old_N) * (self.gamma-2) * self.mean_degree / ((self.gamma-1) * (1-old_N**exponent))
            
            self.HypR = 2* np.log(old_N/(np.pi * mu*k_0*k_0))


        edges = [(i, j, self.J) for i, j in G.edges()]
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
        else:
            epsilon = np.zeros(self.N)
        
            
        return edges, epsilon
  
  
  
"""
Rewiring SD to get to the configuration
"""
class configuration(DeterministicGraph):

    def __init__(self, N, J=1.0, gamma=3, noise=0):
        label = f"configuration model_N{N}_G{gamma}"
        self.gamma = gamma
        super().__init__(N, J, noise=noise, label=label)

    def _generate_graph(self):
        #generate a sequence of random numbers
        degrees = nx.utils.powerlaw_sequence(self.N, exponent=self.gamma)
        sequence = [max(2, int(round(d))) for d in degrees]
        #sequence = [int(round(d)) for d in degrees]
        if sum(sequence) % 2 != 0:
            sequence[0] += 1

        # 2. Generate and Clean Graph
        G_multi = nx.configuration_model(sequence)
        G = nx.Graph(G_multi) # Remove parallel edges
        G.remove_edges_from(nx.selfloop_edges(G))

        # 3. SELECT LARGEST CONNECTED COMPONENT
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
        
        G = nx.convert_node_labels_to_integers(G)
        
        # Update internal N to reflect LCC size
        N_real = G.number_of_nodes()
        print(f"LCC Size: {N_real}/{self.N}")
        self.N=N_real

        # 4. Metrics & Layout
        degrees = [d for n, d in G.degree()]
        max_deg = np.max(degrees)
        avg_deg = np.mean(degrees)
        
        print(max_deg, avg_deg)
        
        pos_dict = nx.spring_layout(G)
        # Handle the fact that nodes in LCC might not be 0 to N-1
        node_list = list(G.nodes())
        self.positions = np.array([pos_dict[i] for i in node_list])
        
        # 5. Physics Parameters
        edges = [(u, v, self.J) for u, v in G.edges()]
        epsilon = np.random.uniform(-self.noise, self.noise, size=self.N) if self.noise != 0 else np.zeros(self.N)
        
        return edges, epsilon
        
        
        
class barabasi(DeterministicGraph):

    def __init__(self, N, J=1.0, m=1, noise=0):
        label = f"barabasi_albert_N{N}_m{m}"
        self.m = m
        super().__init__(N, J, noise=noise, label=label)

    def _generate_graph(self):
        #generate a sequence of random numbers
        G = nx.barabasi_albert_graph(self.N, m=self.m)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
        G = nx.convert_node_labels_to_integers(G)
        # Update internal N to reflect LCC size
        N_real = G.number_of_nodes()
        print(f"LCC Size: {N_real}/{self.N}")
        self.N=N_real

        degrees = [d for n, d in G.degree()]
        max_deg = np.max(degrees)
        avg_deg = np.mean(degrees)
        
        print(max_deg, avg_deg)
        #self.label += f" (k_max={max_deg}, k_avg={avg_deg:.2f})"

        pos_dict = nx.spring_layout(G)
        self.positions = np.array([pos_dict[i] for i in range(self.N)])
        
        edges = [(i, j, self.J) for i, j in G.edges()]
        if self.noise!=0:
            epsilon = np.random.uniform(-self.noise, self.noise, size=self.N)
        else:
            epsilon = np.zeros(self.N)
        return edges, epsilon

