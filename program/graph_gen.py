import numpy as np
import re


# ==========================================================
# Core builders (fast where possible)
# ==========================================================

def _chain_graph(N, J=1.0, periodic=False):
    edges = [(i, i+1, J) for i in range(N-1)]
    if periodic and N > 2:
        edges.append((0, N-1, J))
    
    positions = _chain_coords(N, periodic=periodic)
    return edges, positions


def _chain_range_graph(N, R=1, J=1.0, periodic=False):
    edges = []
    for i in range(N):
        for r in range(1, R+1):
            j = i + r
            if periodic:
                j %= N
            elif j >= N:
                continue
            edges.append((i, j, J))
    
    positions = _chain_coords(N, periodic=periodic)
    return edges, positions
    

def _chain_coords(N, periodic=False):
    if not periodic:
        # Standard linear positions
        return np.column_stack([np.arange(N), np.zeros(N)])
    else:
        # Map to a circle for PBC visualization
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])


# ---------- FAST SQUARE ----------
def _square_graph(Lx, Ly, J=1.0, periodic=False):

    N = Lx * Ly
    grid = np.arange(N).reshape(Ly, Lx)
    edges = []

    edges.extend(zip(grid[:, :-1].ravel(),
                     grid[:, 1:].ravel(),
                     [J]*((Lx-1)*Ly)))

    edges.extend(zip(grid[:-1, :].ravel(),
                     grid[1:, :].ravel(),
                     [J]*(Lx*(Ly-1))))

    if periodic:
        edges.extend(zip(grid[:, -1], grid[:, 0], [J]*Ly))
        edges.extend(zip(grid[-1, :], grid[0, :], [J]*Lx))
        
    x = np.arange(Lx)
    y = np.arange(Ly)
    X, Y = np.meshgrid(x,y)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    return list(edges), positions


# ---------- FAST CUBIC ----------
def _cubic_graph(Lx, Ly, Lz, J=1.0, periodic=False):

    N = Lx * Ly * Lz
    grid = np.arange(N).reshape(Lz, Ly, Lx)
    edges = []

    edges.extend(zip(grid[:, :, :-1].ravel(),
                     grid[:, :, 1:].ravel(),
                     [J]*((Lx-1)*Ly*Lz)))

    edges.extend(zip(grid[:, :-1, :].ravel(),
                     grid[:, 1:, :].ravel(),
                     [J]*(Lx*(Ly-1)*Lz)))

    edges.extend(zip(grid[:-1, :, :].ravel(),
                     grid[1:, :, :].ravel(),
                     [J]*(Lx*Ly*(Lz-1))))

    if periodic:
        edges.extend(zip(grid[:, :, -1].ravel(),
                         grid[:, :, 0].ravel(),
                         [J]*(Ly*Lz)))
        edges.extend(zip(grid[:, -1, :].ravel(),
                         grid[:, 0, :].ravel(),
                         [J]*(Lx*Lz)))
        edges.extend(zip(grid[-1, :, :].ravel(),
                         grid[0, :, :].ravel(),
                         [J]*(Lx*Ly)))

    return list(edges)


# ---------- HYPERBOLIC {p,q} ----------
def _tiling_graph(p, q, n_layers=8, J=1.0):
    from hypertiling import HyperbolicTiling
    from hypertiling.graphics.plot import plot_tiling
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    

    T = HyperbolicTiling(p, q, n_layers)
    colors = np.random.rand(len(T))
    
    # 1. Extract unique vertices and map them to indices
    unique_verts = {} # key: complex coordinate, value: index
    positions_list = []
    edges = set()
    
    # 2. Iterate through polygons to build the vertex-based graph
    for i in range(len(T)):
        # get_vertices returns the corners of the i-th polygon
        poly_coords = T.get_vertices(i)
        
        poly_indices = []
        for v in poly_coords:
            # Rounding is CRITICAL due to floating point precision
            v_rounded = np.round(v.real, 10) + 1j * np.round(v.imag, 10)
            
            if v_rounded not in unique_verts:
                unique_verts[v_rounded] = len(positions_list)
                positions_list.append(v)
            
            poly_indices.append(unique_verts[v_rounded])
        
        # 3. Create edges along the perimeter of this polygon
        n_p = len(poly_indices)
        for k in range(n_p):
            v1 = poly_indices[k]
            v2 = poly_indices[(k + 1) % n_p] # Wrap around to close the polygon
            
            # Sort to ensure (v1, v2) is the same as (v2, v1)
            edge = tuple(sorted((v1, v2)))
            edges.add(edge)

    # Convert to final formats
    N = len(positions_list)
    positions = np.column_stack(([v.real for v in positions_list],
                                 [v.imag for v in positions_list]))
    
    # Add weight J to edges
    weighted_edges = [(int(e[0]), int(e[1]), J) for e in edges]
    
    colors = np.random.rand(len(T))
    plot_tiling(T, colors, cmap=cmap.RdBu, edgecolor="k", linewidth=0.2);
    plt.savefig(f"../figures/hyper_p{p}_q{q}.pdf")
    plt.close()
    return weighted_edges, N, positions



# ==========================================================
# Unified Engine
# ==========================================================

def lattice_graph(kind, N=None, J=1.0, periodic=False, **params):
    """
    Unified lattice generator.

    Returns:
        edges, epsilon
    """

    kind = kind.lower()
    epsilon = np.zeros(N) if N is not None else None

    # ---------------- CHAIN ----------------
    if kind == "chain":
        edges = _chain_graph(N, J, periodic)

    elif kind == "range_chain":
        R = params.get("R", 1)
        edges = _chain_range_graph(N, R, J, periodic)

    # ---------------- SQUARE ----------------
    elif kind == "square":

        if "Lx" in params and "Ly" in params:
            Lx, Ly = params["Lx"], params["Ly"]
        else:
            # auto-infer from N (try square)
            Lx = int(np.sqrt(N))
            Ly = N // Lx

        N = Lx * Ly
        epsilon = np.zeros(N)
        edges = _square_graph(Lx, Ly, J, periodic)

    # ---------------- CUBIC ----------------
    elif kind == "cubic":

        if {"Lx","Ly","Lz"} <= params.keys():
            Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
        else:
            L = int(round(N ** (1/3)))
            Lx = Ly = Lz = L

        N = Lx * Ly * Lz
        epsilon = np.zeros(N)
        edges = _cubic_graph(Lx, Ly, Lz, J, periodic)

    # ---------------- TILING {p,q} ----------------
    elif kind.startswith("{"):

        match = re.match(r"\{(\d+),(\d+)\}", kind)
        if match is None:
            raise ValueError("Use tiling like '{7,3}'")

        p, q = int(match.group(1)), int(match.group(2))
        edges = _tiling_graph(p, q, N, J)

    else:
        raise ValueError(f"Unknown lattice type: {kind}")

    return edges, epsilon
  
  
