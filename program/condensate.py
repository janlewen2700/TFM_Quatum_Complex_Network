import numpy as np
from scipy.optimize import brentq

# ---------- Bose occupation (numerically stable), particles ----------
def bose(eigen, mu, T):
    eigen = np.asarray(eigen)
    x = (eigen - mu) / T

    # small-x expansion to avoid overflow
    small = x < 1e-8
    n = np.empty_like(x)

    n[small] = 1.0 / x[small]
    n[~small] = 1.0 / (np.exp(x[~small]) - 1.0)

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
    
    

def compute_n0_Tc(self, fillings, T_min_search, T_max_search):
    V = len(self._eigenvalues)

    results = {}

    for fill in fillings:
        N = fill * V

        Tc = find_Tc(N, self._eigenvalues, T_min_search, T_max_search)

        T_array = np.linspace(1e-5 * Tc, 1.5 * Tc, 200)

        fractions = []
        for T in T_array:
            mu = solve_mu(N, self._eigenvalues, T)
            n0 = ground_state_occupation(np.min(self._eigenvalues), mu, T)
            fractions.append(n0 / N)

        fractions = np.array(fractions)

        result = {
            "Tc": Tc,
            "T_array": T_array,
            "fractions": fractions,
        }

        # Only fit alpha for fill = 1
        if np.isclose(fill, 1.0):
            alpha, alpha_err = fit_condensate_alpha(
                self,
                fill,
                Tc,
                T_array,
                fractions,
                fit_min_ratio=0.05,
                fit_max_ratio=0.95
            )

            self.Tc = Tc
            self.alpha = alpha

            result["alpha"] = alpha
            result["alpha_err"] = alpha_err

        results[fill] = result

    self.n0_Tc_results = results  # cache it

    return results
    
    
    
def fit_condensate_alpha(self, fill, Tc, T_array, fractions, fit_min_ratio=0.05, fit_max_ratio=0.95):


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
