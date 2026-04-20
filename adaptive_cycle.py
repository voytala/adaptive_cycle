import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp, trapezoid
from numpy.linalg import eigvals
import re


BASE_DIR = Path(__file__).resolve().parent


def resolve_input_path(input_file):
    """
    Plik wejściowy musi leżeć w katalogu BASE_DIR/inputs
    i być podany jako sama nazwa pliku, bez ścieżki.
    """
    if not input_file or not str(input_file).strip():
        raise SystemExit("Błąd: parametr input_file nie może być pusty.")

    input_name = Path(input_file)

    if input_name.is_absolute() or len(input_name.parts) != 1:
        raise SystemExit(
            "Błąd: input_file ma być samą nazwą pliku, "
            "bez ścieżki. Plik musi znajdować się w katalogu 'inputs'."
        )

    inputs_dir = BASE_DIR / "inputs"
    if not inputs_dir.is_dir():
        raise SystemExit(
            f"Błąd: nie znaleziono katalogu wejściowego '{inputs_dir}'."
        )

    input_path = inputs_dir / input_name.name
    if not input_path.is_file():
        raise SystemExit(
            f"Błąd: nie znaleziono pliku wejściowego '{input_name.name}' w katalogu 'inputs'."
        )

    return input_path


def prepare_output_dir(save_prefix):
    """
    Tworzy katalog wynikowy BASE_DIR/save_prefix.
    Jeśli istnieje, usuwa go i tworzy od nowa.
    """
    if save_prefix is None or not str(save_prefix).strip():
        raise SystemExit("Błąd: parametr --save-prefix nie może być pusty.")

    out_name = Path(str(save_prefix).strip())

    if out_name.is_absolute() or len(out_name.parts) != 1 or out_name.name in {".", "..", "inputs"}:
        raise SystemExit(
            "Błąd: --save-prefix ma być pojedynczą nazwą katalogu "
            "(bez ścieżki i bez nazw typu '.' , '..' albo 'inputs')."
        )

    output_dir = BASE_DIR / out_name.name

    if output_dir.exists():
        if output_dir.is_file():
            raise SystemExit(
                f"Błąd: istnieje już plik o nazwie '{output_dir.name}', "
                "nie można utworzyć katalogu wynikowego."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(filename):
    """
    Robust parser for:
        dx_i = x_i ( ... multiline polynomial ... )
    """

    eq_pattern = re.compile(
        r"^dx_(\d+)\s*=\s*x_(\d+)\s*\(\s*(.*?)\s*\)\s*$",
        re.DOTALL
    )

    mono_pattern = re.compile(r"x_(\d+)(?:\^(\d+))?")

    def join_equations(lines):
        """
        Składa multiline equation blocks poprawnie:
        - ignoruje puste linie
        - ignoruje komentarze
        - zachowuje pełne nawiasy
        """
        blocks = []
        buf = ""
        depth = 0

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("dx_") and depth == 0:
                if buf:
                    blocks.append(buf.strip())
                    buf = ""

            buf += " " + line

            depth += line.count("(") - line.count(")")

            if depth == 0 and buf:
                blocks.append(buf.strip())
                buf = ""

        if buf.strip():
            blocks.append(buf.strip())

        if depth != 0:
            raise ValueError("Unbalanced parentheses in model input")

        return blocks

    def split_terms(poly):
        """
        poprawne splitowanie: + i - tylko na poziomie top-level
        """
        poly = poly.replace(" ", "")
        terms = []
        start = 0
        depth = 0

        for i in range(1, len(poly)):
            if poly[i] == "(":
                depth += 1
            elif poly[i] == ")":
                depth -= 1

            if depth == 0 and poly[i] in "+-":
                terms.append(poly[start:i])
                start = i

        terms.append(poly[start:])
        return [t for t in terms if t]

    def parse_term(term, n):
        term = term.replace(" ", "")
        matches = list(mono_pattern.finditer(term))
        exponents = [0] * n

        if matches:
            coeff_part = term[:matches[0].start()]

            for m in matches:
                idx = int(m.group(1)) - 1
                exp = int(m.group(2)) if m.group(2) else 1
                exponents[idx] += exp
        else:
            coeff_part = term

        if coeff_part in ("", "+"):
            coeff = 1.0
        elif coeff_part == "-":
            coeff = -1.0
        else:
            coeff = float(coeff_part)

        return tuple(exponents), coeff

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [l.rstrip("\n") for l in lines if l.strip() and not l.strip().startswith("#")]

    n = int(lines[0])
    x0 = np.array(list(map(float, lines[1].split())))

    eq_blocks = join_equations(lines[2:])

    if len(eq_blocks) != n:
        raise ValueError(f"Expected {n} equations, got {len(eq_blocks)}")

    interactions = {}
    seen = set()

    for block in eq_blocks:
        m = eq_pattern.match(block)
        if not m:
            raise ValueError(f"Cannot parse block:\n{block}")

        i = int(m.group(1))
        j = int(m.group(2))
        poly = m.group(3)

        if i != j:
            raise ValueError("dx_i mismatch")

        seen.add(i)

        eq_terms = defaultdict(float)
        for term in split_terms(poly):
            mon, coeff = parse_term(term, n)
            eq_terms[mon] += coeff

        for mon, coeff in eq_terms.items():
            if mon not in interactions:
                interactions[mon] = [0.0] * n
            interactions[mon][i - 1] += coeff

    if len(seen) != n:
        raise ValueError("Missing equations")

    support_map = defaultdict(list)
    for j in interactions:
        supp = frozenset(i for i, v in enumerate(j) if v != 0)
        support_map[supp].append(j)

    return n, x0, interactions, support_map

def x_pow_j(x, j):
    x = np.asarray(x)
    return np.prod([
        (x[i] ** j[i]) if j[i] != 0 else 1.0
        for i in range(len(x))
    ])


def T_ji(x, interactions, support_map, n):
    """
    Zwraca słownik T[(K,i)] = T^t_{K->i}(x) dla każdej agregowanej interakcji K=frozenset(...)
    """
    T = {}
    for K, js in support_map.items():
        for i in range(n):
            if x[i] > 0:
                total = 0.0
                for j in js:
                    total += interactions[j][i] * x_pow_j(x, j)
                T[(K, i)] = total
            else:
                T[(K, i)] = 0.0
    return T


def f(x, interactions, support_map, n):
    T = T_ji(x, interactions, support_map, n)
    return np.array([x[i] * sum(T[(K, i)] for K in support_map) for i in range(n)])


def g_per_capita(x, interactions, n):
    g = np.zeros(n)
    for i in range(n):
        if x[i] == 0:
            g[i] = 0.0
        else:
            total = 0.0
            for j, alpha in interactions.items():
                total += alpha[i] * x_pow_j(x, j)
            g[i] = total
    return g


def potential(x, interactions, support_map, n):
    x = np.maximum(x, 1e-12)
    T = T_ji(x, interactions, support_map, n)
    H_i = []
    for i in range(n):
        contributions = [abs(T[(K, i)]) for K in support_map]
        Z = sum(contributions)
        if Z == 0:
            H_i.append(0.0)
        else:
            p = [c / Z for c in contributions]
            H_i.append(-Z * sum(pi * np.log(pi) for pi in p if pi > 0))
    H = sum(H_i)
    if H == 0:
        return 0.0
    weights = [hi / H for hi in H_i]
    return -sum(hi * np.log(w) for hi, w in zip(H_i, weights) if w > 0)


def potential_global(x, interactions, support_map, n):
    x = np.maximum(x, 1e-12)
    T = T_ji(x, interactions, support_map, n)
    contribs = [abs(v) for v in T.values()]
    T_total = sum(contribs)
    if T_total == 0:
        return 0.0
    return -sum(a * np.log(a / T_total) for a in contribs if a > 0)


def connectedness(x, interactions, support_map, n):
    x = np.maximum(x, 1e-12)
    T = T_ji(x, interactions, support_map, n)
    T_out = {K: 0.0 for K in support_map}
    T_in = {i: 0.0 for i in range(n)}
    for (K, i), v in T.items():
        a = abs(v)
        T_out[K] += a
        T_in[i] += a

    T_total = sum(T_out.values())
    if T_total == 0:
        return 0.0

    C = 0.0
    for (K, i), v in T.items():
        a = abs(v)
        if a > 0 and T_out[K] > 0 and T_in[i] > 0:
            C += a * np.log((a * T_total) / (T_out[K] * T_in[i]))
    return C


def row_entropy_connectedness(x, interactions, support_map, n):
    x = np.maximum(x, 1e-12)
    T = T_ji(x, interactions, support_map, n)
    T_out = {K: 0.0 for K in support_map}
    for (K, i), v in T.items():
        T_out[K] += abs(v)

    T_total = sum(T_out.values())
    if T_total == 0:
        return 0.0

    H_row = {}
    for K, total_K in T_out.items():
        if total_K == 0:
            H_row[K] = 0.0
        else:
            vals = [abs(T[(K, i)]) for i in range(n)]
            p = [v / total_K for v in vals]
            H_row[K] = -sum(pi * np.log(pi) for pi in p if pi > 0)

    return sum(T_out[K] * H_row[K] for K in T_out)


def jacobian(x, interactions, support_map, n):
    h = 1e-6
    fx = f(x, interactions, support_map, n)
    D = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += h
        D[:, j] = (f(xp, interactions, support_map, n) - fx) / h
    return D


def jacobian_pc(x, interactions, support_map, n):
    h = 1e-6
    fx = g_per_capita(x, interactions, n)
    D = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += h
        D[:, j] = (g_per_capita(xp, interactions, n) - fx) / h
    return D


########## linearyzacja

def linear_interaction_matrix(x, interactions, support_map, n):
    """
    A_ij = ∂f_i / ∂x_j (Jacobian)
    """
    return jacobian(x, interactions, support_map, n)


def linear_potential(x, interactions, support_map, n):
    """
    Potencjał = - entropia wag wierszy Jacobianu
    """
    A = np.abs(linear_interaction_matrix(x, interactions, support_map, n))
    H = 0.0
    for i in range(n):
        row = A[i]
        Z = np.sum(row)
        if Z == 0:
            continue
        p = row / Z
        p = p[p > 0]
        H -= Z * np.sum(p * np.log(p))
    return H


def linear_connectedness(x, interactions, support_map, n):
    """
    Mutual-information-like structure on Jacobian graph
    """
    A = np.abs(linear_interaction_matrix(x, interactions, support_map, n))
    row_sum = A.sum(axis=1)
    col_sum = A.sum(axis=0)
    total = A.sum()

    if total == 0:
        return 0.0

    C = 0.0
    for i in range(n):
        for j in range(n):
            a = A[i, j]
            if a <= 0:
                continue
            C += a * np.log((a * total) / (row_sum[i] * col_sum[j] + 1e-12))
    return C


# =========================
# LAYOUT
# =========================
def circular_layout(n, radius=1.5):

    if n == 2:
        return np.array([
            [-radius, 0.0],
            [radius, 0.0]
        ])

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([
        [radius * np.cos(a), radius * np.sin(a)]
        for a in angles
    ])


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return v / n


def perp(v):
    return np.array([-v[1], v[0]])


def node_radius(x):
    biomass = np.asarray(x)
    biomass = np.maximum(biomass, 1e-12)
    r = 0.03 + 0.12 * np.log1p(biomass) / (np.log1p(np.max(biomass)) + 1e-12)
    return r


# =========================
# DRAW ONE FRAME
# =========================
def draw_frame(ax, x, pos, A, t):

    ax.clear()
    n = len(pos)

    max_w = np.max(np.abs(A)) + 1e-12
    sep = 0.10

    # =====================
    # NODES
    # =====================
    biomass = np.asarray(x)
    biomass = np.maximum(biomass, 1e-12)
    node_sizes = 200 + 900 * np.log1p(biomass) / (np.log1p(np.max(biomass)) + 1e-12)

    radii = node_radius(x)

    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=node_sizes,
        c="black",
        edgecolors="white",
        linewidths=1.2,
        zorder=3
    )

    for i in range(n):
        ax.text(
            pos[i, 0], pos[i, 1],
            str(i + 1),
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            zorder=4
        )

    # =====================
    # EDGES
    # =====================
    for i in range(n):
        for j in range(i + 1, n):

            v = pos[i] - pos[j]
            direction = normalize(v)
            offset_dir = perp(direction)

            # -------- i -> j --------
            w_ij = A[i, j]
            if abs(w_ij) > 1e-6:
                offset = sep * offset_dir

                p1 = pos[i] - radii[i] * direction + offset
                p2 = pos[j] + radii[j] * direction + offset

                ax.annotate(
                    "",
                    xy=p2,
                    xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=("green" if w_ij > 0 else "red"),
                        lw=1.0 + 5.0 * abs(w_ij) / max_w,
                        mutation_scale=12,
                        alpha=0.9
                    ),
                    zorder=2
                )

            # -------- j -> i --------
            w_ji = A[j, i]
            if abs(w_ji) > 1e-6:
                offset = -sep * offset_dir

                p1 = pos[j] - radii[j] * (-direction) + offset
                p2 = pos[i] + radii[i] * (-direction) + offset

                ax.annotate(
                    "",
                    xy=p2,
                    xytext=p1,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=("green" if w_ji > 0 else "red"),
                        lw=1.0 + 5.0 * abs(w_ji) / max_w,
                        mutation_scale=12,
                        alpha=0.9
                    ),
                    zorder=2
                )

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)

    ax.set_title(f"Linearized interaction network (t={t:.2f})")
    ax.set_axis_off()
    ax.set_aspect("equal")


# =========================
# ANIMATION SAVE
# =========================
def save_linear_graph_animation(sol, x_vals, interactions, support_map, n, output_dir):

    pos = circular_layout(n)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    def update(frame):
        x = x_vals[frame]
        A = jacobian(x, interactions, support_map, n)
        draw_frame(ax, x, pos, A, sol.t[frame])

    anim = FuncAnimation(fig, update, frames=len(sol.t), interval=40)

    output_mp4 = output_dir / "linear-graph.mp4"
    output_gif = output_dir / "linear-graph.gif"

    try:
        anim.save(str(output_mp4), fps=20, dpi=180)
        print(f"Saved: {output_mp4}")
    except Exception:
        anim.save(str(output_gif), writer=PillowWriter(fps=12))
        print(f"Saved: {output_gif}")

    plt.close(fig)


########## rzeczy do rysowania hipergrafu


def compute_layout(n, support_map):
    pos = {}

    for i in range(n):
        pos[f"x{i}"] = (i, 0)

    for idx, K in enumerate(support_map):
        pos[f"K{tuple(sorted(K))}"] = (idx, 1)

    return pos


def draw_hypergraph(ax, x, interactions, support_map, n):
    ax.clear()

    T = T_ji(x, interactions, support_map, n)

    max_w = max([abs(v) for v in T.values()] + [1e-12])

    for (K, i), v in T.items():
        if abs(v) < 1e-12:
            continue

        w = np.log1p(abs(v)) / np.log1p(max_w)
        color = "green" if v > 0 else "red"
        alpha = 0.2 + 0.8 * w
        lw = 0.5 + 4.0 * w

        for j in K:
            ax.plot(
                [j, i],
                [1, 0],
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle='round'
            )

    ax.scatter(
        range(n), [0] * n,
        s=250,
        c="black",
        edgecolors="white",
        linewidths=1.2,
        zorder=3
    )

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks([])
    ax.set_title("Dynamic hypergraph of interactions T_ji(x)", fontsize=12)
    ax.grid(alpha=0.2)


def interactive_hypergraph(sol, x_vals, interactions, support_map, n):
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.subplots_adjust(bottom=0.25)

    t_idx = 0
    draw_hypergraph(ax, x_vals[t_idx], interactions, support_map, n)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 't', 0, len(sol.t) - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        draw_hypergraph(ax, x_vals[i], interactions, support_map, n)
        ax.set_title(f"t = {sol.t[i]:.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def save_hypergraph_animation(sol, x_vals, interactions, support_map, n, output_dir):
    fig, ax = plt.subplots(figsize=(9, 4))

    try:
        import matplotlib.animation as animation

        def frame(i):
            draw_hypergraph(ax, x_vals[i], interactions, support_map, n)
            ax.set_title(f"t = {sol.t[i]:.3f}")

        anim = animation.FuncAnimation(
            fig,
            frame,
            frames=len(sol.t),
            interval=50
        )

        output = output_dir / "hypergraph.mp4"
        anim.save(str(output), fps=15)
        print(f"Saved: {output}")

    except Exception as e:
        print("MP4 failed → saving GIF instead", e)
        output = output_dir / "hypergraph.gif"
        anim.save(str(output), writer=PillowWriter(fps=10))
        print(f"Saved: {output}")

    plt.close(fig)


###### odporności

def R_loc(x, interactions, support_map, n):
    vals = eigvals(jacobian(x, interactions, support_map, n))
    return -np.max(vals.real)


def R_loc_pc(x, interactions, support_map, n):
    vals = eigvals(jacobian_pc(x, interactions, support_map, n))
    return -np.max(vals.real)


def R_spec(x, interactions, support_map, n):
    J = jacobian(x, interactions, support_map, n)
    A = np.abs(J)
    Dout = np.diag(A.sum(axis=1))
    Din = np.diag(A.sum(axis=0))
    c = 1.0 / np.max(A) if np.max(A) > 0 else 1.0

    def inv_sqrt(M):
        if np.allclose(M, 0):
            return np.zeros_like(M)
        diag = np.diag(M)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_diag = np.where(diag > 0, 1 / np.sqrt(diag), 0.0)
        return np.diag(inv_sqrt_diag)

    Lout = c * inv_sqrt(Dout) @ (Dout - A) @ inv_sqrt(Dout)
    Lin = c * inv_sqrt(Din) @ (Din - A) @ inv_sqrt(Din)
    vals = np.concatenate([eigvals(Lout), eigvals(Lin)])
    nonzero = [v.real for v in vals if abs(v) > 1e-8]
    return min(abs(v) for v in nonzero) if nonzero else 0.0


def R_spec_pc(x, interactions, support_map, n):
    J = jacobian_pc(x, interactions, support_map, n)
    A = np.abs(J)
    Dout = np.diag(A.sum(axis=1))
    Din = np.diag(A.sum(axis=0))
    c = 1.0 / np.max(A) if np.max(A) > 0 else 1.0

    def inv_sqrt(M):
        if np.allclose(M, 0):
            return np.zeros_like(M)
        diag = np.diag(M)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_diag = np.where(diag > 0, 1 / np.sqrt(diag), 0.0)
        return np.diag(inv_sqrt_diag)

    Lout = c * inv_sqrt(Dout) @ (Dout - A) @ inv_sqrt(Dout)
    Lin = c * inv_sqrt(Din) @ (Din - A) @ inv_sqrt(Din)
    vals = np.concatenate([eigvals(Lout), eigvals(Lin)])
    nonzero = [v.real for v in vals if abs(v) > 1e-8]
    return min(abs(v) for v in nonzero) if nonzero else 0.0


def R_energy_time_series(t, x_vals, interactions, support_map, n):
    """
    Zwraca tablicę R_energy(t_i) = - ∫_{t_i}^{t_{i+τ}} ||f(x(s))||^2 ds
    Zakładamy równomierną siatkę czasową i stałe τ (np. 1 jednostka czasu).
    """
    tau = 1.0
    R_vals = []
    for i, t_i in enumerate(t):
        t_max = t_i + tau
        j = i
        while j < len(t) and t[j] <= t_max:
            j += 1
        if j == i:
            R_vals.append(0.0)
            continue
        norms_sq = [np.linalg.norm(f(x_vals[k], interactions, support_map, n))**2 for k in range(i, j)]
        integral = trapezoid(norms_sq, t[i:j])
        R_vals.append(-integral)
    return np.array(R_vals)


def R_energy_log_time_series(t, x_vals, interactions, support_map, n):
    """
    Zwraca tablicę R_energy^{log}(t_i) = - ∫_{t_i}^{t_{i+τ}} ||f(log x(s))||^2 ds
    Zakładamy równomierną siatkę czasową i stałe τ (np. 5 jednostek czasu).
    """
    tau = 5.0
    R_vals = []
    for i, t_i in enumerate(t):
        t_max = t_i + tau
        j = i
        while j < len(t) and t[j] <= t_max:
            j += 1
        if j == i:
            R_vals.append(0.0)
            continue
        norms_sq = []
        for k in range(i, j):
            x_log = np.log(np.abs(x_vals[k]) + 1e-12)
            norms_sq.append(np.linalg.norm(f(x_log, interactions, support_map, n))**2)
        integral = trapezoid(norms_sq, t[i:j])
        R_vals.append(-integral)
    return np.array(R_vals)


def R_norm(x, interactions, support_map, n):
    return -np.linalg.norm(f(x, interactions, support_map, n))**2


def R_spec_hyper(x, interactions, n):
    A = np.zeros((n, n))
    for j, alpha in interactions.items():
        xj = x_pow_j(x, j)
        for k in range(n):
            if j[k] > 0:
                for i in range(n):
                    A[k, i] += abs(alpha[i] * xj)

    Dout = np.diag(A.sum(axis=1))
    Din = np.diag(A.sum(axis=0))

    A_max = np.max(A)
    c = 1.0 / A_max if A_max > 0 else 1.0

    def safe_inv_sqrt_diag(D):
        return np.diag([1 / np.sqrt(v) if v > 0 else 0.0 for v in np.diag(D)])

    Dout_inv_sqrt = safe_inv_sqrt_diag(Dout)
    Din_inv_sqrt = safe_inv_sqrt_diag(Din)

    Lout = c * Dout_inv_sqrt @ (Dout - A) @ Dout_inv_sqrt
    Lin = c * Din_inv_sqrt @ (Din - A.T) @ Din_inv_sqrt

    eigvals_combined = np.concatenate([np.linalg.eigvals(Lout), np.linalg.eigvals(Lin)])
    nonzero_real_parts = [abs(ev.real) for ev in eigvals_combined if abs(ev) > 1e-8]

    return min(nonzero_real_parts) if nonzero_real_parts else 0.0


def plot_phase_colored_curve(potential_vals, connected_vals, resilience_vals, time_vals,
                             xlabel="Potential", ylabel="Connectedness",
                             rlabel="Resilience", cmap="viridis",
                             vmin=None, vmax=None,
                             tmin=None, tmax=None):
    """
    Rysuje trajektorię (potential vs connectedness) jako krzywą,
    gdzie kolor odcinka odpowiada wartości resilience.
    """
    mask = np.ones_like(time_vals, dtype=bool)
    if tmin is not None:
        mask &= time_vals >= tmin
    if tmax is not None:
        mask &= time_vals <= tmax

    pot = potential_vals[mask]
    conn = connected_vals[mask]
    res = resilience_vals[mask]
    tsel = time_vals[mask]

    if len(pot) < 2:
        raise ValueError("Za mało punktów w wybranym zakresie czasu do narysowania krzywej.")

    fig, ax = plt.subplots(figsize=(6, 5))

    points = np.array([pot, conn]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if vmin is None:
        vmin = res.min()
    if vmax is None:
        vmax = res.max()

    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(res)
    lc.set_linewidth(2)

    line = ax.add_collection(lc)

    ax.scatter(pot[0], conn[0],
               color="green", edgecolor="black", s=80, zorder=5,
               label=f"(t={tsel[0]:.2f})")
    ax.scatter(pot[-1], conn[-1],
               color="red", edgecolor="black", s=80, zorder=5,
               label=f"(t={tsel[-1]:.2f})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Adaptacyjny cykl w regulowanym układzie troficznym")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label(rlabel)

    ax.set_xlim(pot.min(), pot.max())
    ax.set_ylim(conn.min(), conn.max())
    ax.legend(loc="best")

    return fig


# =========================
# PLOTS (STATIC)
# =========================

# Tu ustawiasz, które wykresy trafiają do których plików.
# Zmieniasz tylko ten słownik.
PLOT_GROUPS = {
    "linearized_plots": [
        "biomass",
        "linear_potential",
        "linear_connectedness",
        "R_loc",
        "R_energy",
        "R_spec",
    ],
    "hypergraph_plots": [
        "biomass",
        "potential",
        "potential_global",
        "connectedness",
        "R_energy",
        "Rspec_hyper",
    ],
}

# Etykiety i styl dla pojedynczych serii
PLOT_LABELS = {
    "potential": "Φ: Local Potential",
    "potential_global": "Ψ: Global Entropy",
    "connectedness": "Connectedness",
    "R_loc": "R_loc",
    "R_spec": "R_spec",
    "R_energy": "R_energy",
    "Rspec_hyper": "R_spec_hyper",
    "linear_potential": "Linear Potential",
    "linear_connectedness": "Linear Connectedness",
}

PLOT_STYLES = {
    "potential": "g-",
    "potential_global": "g-",
    "connectedness": "b-",
    "R_loc": "r-",
    "R_loc_pc": "r-",
    "R_spec": "r-",
    "R_energy": "r-",
    "Rspec_hyper": "r-",
    "linear_potential": "g-",
    "linear_connectedness": "b-",
}


def plot_group(sol, data_dict, group_keys, group_name=None,
               hspace=0.06, top=0.95, bottom=0.08, left=0.12, right=0.85):
    """
    Rysuje jedną grupę wykresów jako osobną figurę.
    data_dict:
        key -> 1D array albo lista 1D arrayów
    """
    n_plots = len(group_keys)
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(7.0, 2.2 * n_plots),
        sharex=True,
        gridspec_kw={'hspace': hspace}
    )

    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, group_keys):
        y_data = data_dict[key]

        if key == "biomass":
            # biomass = lista serii, po jednej na gatunek
            for i, y in enumerate(y_data):
                ax.plot(sol.t, y, label=f"Gatunek {i+1}")
            ylabel = "Biomasa"
        else:
            # zwykła pojedyncza seria
            style = PLOT_STYLES.get(key, None)
            label = PLOT_LABELS.get(key, key)

            if style is None:
                ax.plot(sol.t, y_data, label=label)
            else:
                ax.plot(sol.t, y_data, style, label=label)

            ylabel = label

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

        # sensowne limity osi Y
        if key == "biomass":
            all_vals = np.concatenate([np.asarray(y) for y in y_data])
        else:
            all_vals = np.asarray(y_data)

        ymin = np.min(all_vals)
        ymax = np.max(all_vals)
        center = (ymax + ymin) / 2
        half_range = max(ymax - center, center - ymin)
        margin = 0.04 * (2 * half_range if half_range != 0 else 1.0)
        ax.set_ylim(center - half_range - margin, center + half_range + margin)

        if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
            ax.axhline(0, color='grey', linewidth=0.5, alpha=0.5)

        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    axes[-1].set_xlabel("Czas", fontsize=12)

    if group_name is not None:
        fig.suptitle(group_name, fontsize=12)

    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace)
    return fig


def save_all_figures(output_dir, sol, data_dict):
    """
    Zapisuje każdą grupę wykresów do osobnego pliku.
    Nazwa pliku = klucz z PLOT_GROUPS, np. main.pdf, resilience.pdf
    """
    for group_name, group_keys in PLOT_GROUPS.items():
        fig = plot_group(sol, data_dict, group_keys, group_name=group_name)
        fig.savefig(output_dir / f"{group_name}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{group_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--time', type=float, default=150)
    parser.add_argument('--save-prefix', type=str, default=None)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_file)

    if args.save_prefix is not None and not str(args.save_prefix).strip():
        raise SystemExit("Błąd: parametr --save-prefix nie może być pusty.")

    output_dir = None
    if args.save_prefix:
        output_dir = prepare_output_dir(args.save_prefix)
        print(f"Zapis wyników do katalogu: {output_dir}")

    n, x0, interactions, support_map = load_model(input_path)

    t_eval = np.linspace(0, args.time, int(args.time * 10))
    def rhs(t, x):
        x = np.clip(x, 1e-12, 1e6)  # stabilizacja
        return f(x, interactions, support_map, n)

    sol = solve_ivp(
        rhs,
        (0, args.time),
        x0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-7,
        atol=1e-9,
        max_step=0.1
    )

    x_vals = sol.y.T

    # =========================
    # METRICS
    # =========================
    potential_vals = np.array([potential(x, interactions, support_map, n) for x in x_vals])
    potential_global_vals = np.array([potential_global(x, interactions, support_map, n) for x in x_vals])
    connected_vals = np.array([connectedness(x, interactions, support_map, n) for x in x_vals])
    row_entropy_vals = np.array([row_entropy_connectedness(x, interactions, support_map, n) for x in x_vals])

    Rloc = np.array([R_loc(x, interactions, support_map, n) for x in x_vals])
    Rloc_pc = np.array([R_loc_pc(x, interactions, support_map, n) for x in x_vals])

    Rspec = np.array([R_spec(x, interactions, support_map, n) for x in x_vals])
    Rspec_pc = np.array([R_spec_pc(x, interactions, support_map, n) for x in x_vals])

    Renergy = R_energy_time_series(sol.t, x_vals, interactions, support_map, n)
    Renergy_pc = R_energy_log_time_series(sol.t, x_vals, interactions, support_map, n)

    Rnorm = np.array([R_norm(x, interactions, support_map, n) for x in x_vals])
    Rspec_hyper = np.array([R_spec_hyper(x, interactions, n) for x in x_vals])

    linear_pot_vals = np.array([
        linear_potential(x, interactions, support_map, n)
        for x in x_vals
    ])

    linear_conn_vals = np.array([
        linear_connectedness(x, interactions, support_map, n)
        for x in x_vals
    ])

    # =========================
    # SAVE CSV
    # =========================
    if output_dir is not None:
        np.savetxt(
            str(output_dir / "potential.csv"),
            np.column_stack((sol.t, potential_vals)),
            delimiter=",", header="time,potential", comments=""
        )
        np.savetxt(
            str(output_dir / "potential_global.csv"),
            np.column_stack((sol.t, potential_global_vals)),
            delimiter=",", header="time,potential_global", comments=""
        )
        np.savetxt(
            str(output_dir / "connectedness.csv"),
            np.column_stack((sol.t, connected_vals)),
            delimiter=",", header="time,connectedness", comments=""
        )
        #np.savetxt(
        #    str(output_dir / "row_entropy_conn.csv"),
        #    np.column_stack((sol.t, row_entropy_vals)),
        #    delimiter=",", header="time,row_entropy_connectedness", comments=""
        #)
        np.savetxt(
            str(output_dir / "R_loc.csv"),
            np.column_stack((sol.t, Rloc)),
            delimiter=",", header="time,R_loc", comments=""
        )
        #np.savetxt(
        #    str(output_dir / "R_loc_pc.csv"),
        #    np.column_stack((sol.t, Rloc_pc)),
        #    delimiter=",", header="time,R_loc_pc", comments=""
        #)
        np.savetxt(
            str(output_dir / "R_spec.csv"),
            np.column_stack((sol.t, Rspec)),
            delimiter=",", header="time,R_spec", comments=""
        )
        #np.savetxt(
        #    str(output_dir / "R_spec_pc.csv"),
        #    np.column_stack((sol.t, Rspec_pc)),
        #    delimiter=",", header="time,R_spec_pc", comments=""
        #)
        np.savetxt(
            str(output_dir / "R_energy.csv"),
            np.column_stack((sol.t, Renergy)),
            delimiter=",", header="time,R_energy", comments=""
        )
        #np.savetxt(
        #    str(output_dir / "R_energy_pc.csv"),
        #    np.column_stack((sol.t, Renergy_pc)),
        #    delimiter=",", header="time,R_energy_pc", comments=""
        #)
        #np.savetxt(
        #    str(output_dir / "Rnorm.csv"),
        #    np.column_stack((sol.t, Rnorm)),
        #    delimiter=",", header="time,R_norm", comments=""
        #)
        np.savetxt(
            str(output_dir / "Rspec_hyper.csv"),
            np.column_stack((sol.t, Rspec_hyper)),
            delimiter=",", header="time,Rspec_hyper", comments=""
        )
        np.savetxt(
            str(output_dir / "linear_potential.csv"),
            np.column_stack((sol.t, linear_pot_vals)),
            delimiter=",", header="time,linear_potential", comments=""
        )
        np.savetxt(
            str(output_dir / "linear_connectedness.csv"),
            np.column_stack((sol.t, linear_conn_vals)),
            delimiter=",", header="time,linear_connectedness", comments=""
        )
        np.savetxt(
            str(output_dir / "biomass.csv"),
            np.column_stack((sol.t, x_vals)),
            delimiter=",",
            header="time," + ",".join([f"x{i+1}" for i in range(n)]),
            comments=""
        )

    data_dict = {
        "biomass": [sol.y[i] for i in range(n)],
        "potential": potential_vals,
        "potential_global": potential_global_vals,
        "connectedness": connected_vals,
        #"row_entropy_conn": row_entropy_vals,
        "R_loc": Rloc,
        #"R_loc_pc": Rloc_pc,
        "R_spec": Rspec,
        #"R_spec_pc": Rspec_pc,
        "R_energy": Renergy,
        #"R_energy_pc": Renergy_pc,
        #"R_norm": Rnorm,
        "Rspec_hyper": Rspec_hyper,
        "linear_potential": linear_pot_vals,
        "linear_connectedness": linear_conn_vals,
    }

    # =========================
    # PLOTS (STATIC)
    # =========================
    def plot_all(hspace=0.025, top=0.98, bottom=0.06, left=0.15, right=0.85,
                 figsize_per_plot=(6, 1), legend_inside=False, title_fontsize=9):
        plot_data = [
            ([sol.y[i] for i in range(n)], 'G', None, 'Biomasa', None),
            ([potential_vals], 'Φ: Local Potential', None, 'Potencjał Φ', 'g-'),
            ([potential_global_vals], 'Ψ: Global Entropy', None, 'Potencjał Ψ', 'g-'),
            ([connected_vals], 'Connectedness', None, 'Spójność', 'b-'),
            ([Rloc], 'R_loc', None, 'R1', 'r-'),
            ([Rspec], 'R_spec', None, 'R2', 'r-'),
            ([Renergy], 'R_energy', None, 'R3', 'r-'),
            ([Rspec_hyper], 'R_spec_hyper', None, 'R4', 'r-'),
            ([linear_pot_vals], None, None, "Linear Potential", "m-"),
            ([linear_conn_vals], None, None, "Linear Connectedness", "c-"),
        ]

        n_plots = len(plot_data)
        figsize = (figsize_per_plot[0], figsize_per_plot[1] * n_plots)
        fig, axes = plt.subplots(
            n_plots, 1, figsize=figsize, sharex=True,
            gridspec_kw={'hspace': hspace}
        )

        if n_plots == 1:
            axes = [axes]

        for ax, (y_data_list, label, title, ylabel, style) in zip(axes, plot_data):
            for idx, y_data in enumerate(y_data_list):
                if style:
                    ax.plot(sol.t, y_data, style, label=label if idx == 0 else None)
                else:
                    ax.plot(sol.t, y_data, label=f"{label}{idx+1}" if len(y_data_list) > 1 else label)

            ax.set_title(title, fontsize=title_fontsize)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)

            ymin = min(y.min() for y in y_data_list)
            ymax = max(y.max() for y in y_data_list)
            center = (ymax + ymin) / 2
            half_range = max(ymax - center, center - ymin)
            margin = 0.04 * (2 * half_range if half_range != 0 else 1.0)
            ax.set_ylim(center - half_range - margin, center + half_range + margin)

            if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
                ax.axhline(0, color='grey', linewidth=0.5, alpha=0.5)

            if label == 'G':
                ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.35),
                    fontsize=10,
                    ncol=1,
                    borderaxespad=0
                )

        for ax in axes[:-1]:
            ax.label_outer()

        axes[-1].set_xlabel("Czas", fontsize=12)
        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace)
        return fig

    # =========================
    # MAIN OUTPUT PIPELINE
    # =========================
    if output_dir is not None:
        save_all_figures(output_dir, sol, data_dict)

        save_hypergraph_animation(
            sol,
            x_vals,
            interactions,
            support_map,
            n,
            output_dir
        )

        save_linear_graph_animation(
            sol,
            x_vals,
            interactions,
            support_map,
            n,
            output_dir
        )

    else:
        fig = plot_all()
        plt.show()
        interactive_hypergraph(sol, x_vals, interactions, support_map, n)


if __name__ == "__main__":
    main()