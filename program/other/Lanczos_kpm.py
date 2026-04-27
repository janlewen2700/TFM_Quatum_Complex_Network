import numpy as np
from scipy.sparse.linalg import LinearOperator

def lanczos_dos(H, m=200):

    N = H.shape[0]

    # random initial vector
    v = np.random.randn(N)
    v /= np.linalg.norm(v)

    alpha = []
    beta = []

    v_prev = np.zeros(N)
    v_curr = v

    for j in range(m):

        w = H @ v_curr
        a = np.dot(v_curr, w)
        w = w - a * v_curr - (beta[-1] * v_prev if j > 0 else 0)

        b = np.linalg.norm(w)

        alpha.append(a)
        beta.append(b)

        if b < 1e-12:
            break

        v_prev = v_curr
        v_curr = w / b

    # build tridiagonal matrix
    k = len(alpha)
    T = np.zeros((k, k))

    for i in range(k):
        T[i, i] = alpha[i]
        if i < k-1:
            T[i, i+1] = beta[i+1]
            T[i+1, i] = beta[i+1]

    # diagonalise small matrix
    evals, evecs = np.linalg.eigh(T)

    # spectral weights
    weights = evecs[0,:]**2

    return evals, weights
    
    
    def compute_dos(evals, weights, bins=200, eta=0.05):

    E_grid = np.linspace(evals.min(), evals.max(), bins)
    dos = np.zeros_like(E_grid)

    for E, w in zip(evals, weights):
        dos += w * np.exp(-(E_grid - E)**2 / (2*eta**2))

    return E_grid, dos
    
    
#average over multiple runs:
for r in range(n_random):
    evals, weights = lanczos_dos(H)
    ...
    
#KMP for sparse matrices
import kwant.kpm

def kpm_kwant(H, num_moments=1000):

    rho = kwant.kpm.SpectralDensity(H,
                                     num_moments=num_moments)

    energies = rho.energies
    density = rho()

    print(np.sum(density) * dE)
    dE = energies[1] - energies[0]
    idos = np.cumsum(density) * dE
    
    return energies, density, idos


#or:
import numpy as np
from scipy.sparse.linalg import eigsh

def kpm_dos(H, M=300, R=10, bins=400):
    """
    H  : sparse Hamiltonian
    M  : number of Chebyshev moments
    R  : number of random vectors
    """

    N = H.shape[0]

    # ---- Estimate spectral bounds ----
    Emin = eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]
    Emax = eigsh(H, k=1, which='LA', return_eigenvectors=False)[0]

    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    H_rescaled = (H - b * np.eye(N)) / a

    # ---- Chebyshev moments ----
    mu = np.zeros(M)

    for r in range(R):
        v0 = np.random.randn(N)
        v0 /= np.linalg.norm(v0)

        v1 = H_rescaled @ v0
        mu[0] += np.dot(v0, v0)
        mu[1] += np.dot(v0, v1)

        for m in range(2, M):
            v2 = 2 * (H_rescaled @ v1) - v0
            mu[m] += np.dot(v0, v2)
            v0, v1 = v1, v2

    mu /= R

    # ---- Jackson kernel (removes Gibbs oscillations) ----
    n = np.arange(M)
    g = ( (M - n + 1) * np.cos(np.pi*n/(M+1))
          + np.sin(np.pi*n/(M+1)) / np.tan(np.pi/(M+1)) ) / (M+1)

    mu *= g

    # ---- Reconstruct DOS ----
    E = np.linspace(-1, 1, bins)
    dos = np.zeros_like(E)

    for m in range(M):
        dos += mu[m] * np.cos(m * np.arccos(E))

    dos /= (np.pi * np.sqrt(1 - E**2))

    # rescale energy back
    E_real = a * E + b
    dos /= a

    return E_real, dos
