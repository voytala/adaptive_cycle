import numpy as np
from collections import defaultdict


# ============================================================
# Utils
# ============================================================

def x_pow_j(x, j):
    """
    Π x_i^{j_i}
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
# 1. T_ji (JEDYNE źródło dynamiki)
# ============================================================

def T_ji(x, interactions):
    """
    T_{j->i}(t) = Σ α_k→i * x^{[k]}(t)

    Tu każdy monomial jest traktowany jako osobny składnik
    przepływu j→i, ale agregujemy je do jednego T.
    """

    T = {}

    x = np.maximum(x, 1e-12)

    for (monomial, i), coeff in interactions.items():
        sources = tuple(j for j, p in enumerate(monomial) if p > 0)

        if len(sources) == 0:
            continue

        weight = coeff * x_pow_j(x, monomial)

        if abs(weight) > 1e-15:
            key = (sources, i)

            if key in T:
                T[key] += weight
            else:
                T[key] = weight

    return T


# ============================================================
# 2. Hypergraph z T (wspólna struktura dla wszystkiego)
# ============================================================

def extract_hyperedges_from_T(T):
    """
    Jedna struktura dla Laplasjanu i spójności.
    """
    edges = []

    for (sources, target), w in T.items():
        if abs(w) > 1e-15:
            edges.append((sources, target, w))

    return edges


# ============================================================
# 3. Degree computation
# ============================================================

def compute_degrees(edges, n):
    d = np.zeros(n)

    for sources, target, w in edges:
        d[target] += abs(w)

    return d


# ============================================================
# 4. Laplacian action
# ============================================================

def laplacian_action(x, edges, degrees, k):
    n = len(x)
    y = np.zeros(n)

    for sources, target, w in edges:
        prod = 1.0
        for i in sources:
            prod *= x[i]
        y[target] -= w * prod

    for i in range(n):
        y[i] += degrees[i] * (x[i] ** (k - 1))

    return y


# ============================================================
# 5. Power method
# ============================================================

def power_method_hypergraph(edges, n, k, max_iter=80, tol=1e-7):
    degrees = compute_degrees(edges, n)

    x = np.random.rand(n)
    x = normalize(x)

    lam_old = 0.0

    for _ in range(max_iter):
        y = laplacian_action(x, edges, degrees, k)

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
# 6. MAIN SPECTRUM API (NOW CONSISTENT WITH C(t))
# ============================================================

def compute_tensor_laplacian_spectrum(interactions, x, n, k):
    """
    Laplasjan liczony na T_ji — spójny z definicją spójności.
    """

    T = T_ji(x, interactions)
    edges = extract_hyperedges_from_T(T)

    lam, vec = power_method_hypergraph(edges, n, k)

    return {
        "lambda": lam,
        "eigenvector": vec,
        "edges": edges,
        "T": T
    }


# ============================================================
# 7. STATIC VERSION (debug only)
# ============================================================

def compute_static_tensor_laplacian_spectrum(interactions, n, k):
    T = T_ji(np.ones(n), interactions)
    edges = extract_hyperedges_from_T(T)

    lam, vec = power_method_hypergraph(edges, n, k)

    return {
        "lambda": lam,
        "eigenvector": vec,
        "edges": edges
    }