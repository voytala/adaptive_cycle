import numpy as np
from collections import defaultdict

# ============================================================
# Utils
# ============================================================

def x_pow_j(x, j):
    """
    Monomial evaluation: Π x_i^{j_i}
    """
    x = np.asarray(x)
    return np.prod([
        (x[i] ** j[i]) if j[i] != 0 else 1.0
        for i in range(len(x))
    ])


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ============================================================
# 1. Dynamic hyperedge extraction
# ============================================================

def extract_hyperedges(interactions, x=None):
    """
    Converts polynomial interactions into hyperedges.

    If x is provided → dynamic weighting (IMPORTANT CHANGE).
    """

    edges = []

    for monomial, coeff_vec in interactions.items():
        sources = tuple(i for i, p in enumerate(monomial) if p > 0)

        if len(sources) == 0:
            continue

        activation = 1.0 if x is None else x_pow_j(x, monomial)

        for target, w in enumerate(coeff_vec):
            if abs(w) > 1e-15:
                edges.append((sources, target, w * activation))

    return edges


# ============================================================
# 2. Degree computation
# ============================================================

def compute_degrees(edges, n):
    """
    d[i] = sum of incoming hyperedge weights
    """
    d = np.zeros(n)

    for sources, target, w in edges:
        d[target] += abs(w)

    return d


# ============================================================
# 3. Laplacian action (implicit tensor operator)
# ============================================================

def laplacian_action(x, edges, degrees, k):
    """
    L(x) = D(x) - A(x)
    """
    n = len(x)
    y = np.zeros(n)

    # adjacency part
    for sources, target, w in edges:
        prod = 1.0
        for i in sources:
            prod *= x[i]
        y[target] -= w * prod

    # diagonal part
    for i in range(n):
        y[i] += degrees[i] * (x[i] ** (k - 1))

    return y


# ============================================================
# 4. Power method (tensor eigenvalue approximation)
# ============================================================

def power_method_hypergraph(edges, n, k, max_iter=80, tol=1e-7):

    degrees = compute_degrees(edges, n)

    x = np.random.rand(n)
    x = normalize(x)

    lam_old = 0.0

    for _ in range(max_iter):

        y = laplacian_action(x, edges, degrees, k)

        # tensor Rayleigh quotient approximation
        lam_vals = []

        for i in range(n):
            if abs(x[i]) > 1e-12:
                lam_vals.append(y[i] / (x[i] ** (k - 1)))

        lam = np.mean(lam_vals) if lam_vals else 0.0

        x = normalize(y)

        if abs(lam - lam_old) < tol:
            break

        lam_old = lam

    return lam, x


# ============================================================
# 5. Main API (USED BY YOUR PROJECT)
# ============================================================

def compute_tensor_laplacian_spectrum(interactions, x, n, k):
    """
    Fully dynamic tensor Laplacian spectrum.

    IMPORTANT:
        - depends on x(t)
        - recomputed at every time step
    """

    edges = extract_hyperedges(interactions, x)

    lam, vec = power_method_hypergraph(edges, n, k)

    return {
        "lambda": lam,
        "eigenvector": vec,
        "edges": edges
    }


# ============================================================
# 6. Convenience wrapper (for backward compatibility)
# ============================================================

def compute_static_tensor_laplacian_spectrum(interactions, n, k):
    """
    Old version (for debugging / comparison only).
    """

    edges = extract_hyperedges(interactions, x=None)

    lam, vec = power_method_hypergraph(edges, n, k)

    return {
        "lambda": lam,
        "eigenvector": vec,
        "edges": edges
    }