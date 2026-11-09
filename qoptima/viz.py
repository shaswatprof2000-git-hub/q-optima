"""Graphics for Q-Optima: charts, schematics, Bloch spheres and infographics.

Every function returns a matplotlib Figure, so it works in Colab, Jupyter, scripts
(`fig.savefig(...)`) and the `python -m qoptima figures` command.
Colours: a colour-blind-checked categorical palette, used in a fixed order.
"""
from __future__ import annotations

import numpy as np

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
GRAY = "#b8b7b0"


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


def _clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Schematics
# ---------------------------------------------------------------------------
def pipeline_flowchart(ax=None):
    """Infographic flowchart: from a real decision to a checked answer."""
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    steps = [
        ("1  Real problem", "routes, stocks,\nbuyers & plots, networks", BLUE),
        ("2  QUBO", "min  xᵀQx\nconstraints → penalties", BLUE),
        ("3  Ising / qubits", "x = (1 − z)/2\none qubit per decision", BLUE),
        ("4  Quantum circuit", "H · e^(−iγC) · e^(−iβX)\nrepeated p times", ORANGE),
        ("5  Run & measure", "simulator, noise model\nor IBM quantum hardware", ORANGE),
        ("6  Check", "best sample vs proven\nclassical optimum", AQUA),
    ]
    w, h, gap, y = 1.85, 1.9, 0.27, 1.6
    for i, (title, body, col) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=0, facecolor=col, alpha=0.12))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle="square,pad=0",
                                    linewidth=0, facecolor=col))
        ax.text(x + 0.12, y + h - 0.42, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.12, y + 0.72, body, fontsize=9, color=INK2, va="center", linespacing=1.4)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + w + gap - 0.02, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=INK2, linewidth=1.2))
    # feedback loop: classical optimiser trains the angles
    x4 = 0.15 + 3 * (w + gap) + w / 2
    x5 = 0.15 + 4 * (w + gap) + w / 2
    ax.add_patch(FancyArrowPatch((x5, y - 0.05), (x4, y - 0.05), connectionstyle="arc3,rad=-0.45",
                                 arrowstyle="-|>", mutation_scale=12, color=ORANGE, linewidth=1.4))
    ax.text((x4 + x5) / 2, 0.2, "classical optimiser updates γ, β\n(hybrid quantum–classical loop)",
            ha="center", fontsize=9, color=INK2)
    for xx, label, col in [(0.15, "classical", BLUE), (0.15 + 3 * (w + gap), "quantum", ORANGE),
                           (0.15 + 5 * (w + gap), "verification", AQUA)]:
        ax.text(xx, y + h + 0.25, label.upper(), fontsize=8, color=col, fontweight="bold")
    if own:
        ax.set_title("How Q-Optima solves a problem", pad=4)
        return fig
    return ax


def draw_circuit(circuit, max_layers: int = 26, title: str | None = None):
    """Schematic of a (small) circuit: wires, gate boxes, CNOT dots."""
    plt = _plt()
    from matplotlib.patches import Circle, FancyBboxPatch
    layers = circuit.layers()[:max_layers]
    n = circuit.n
    fig, ax = plt.subplots(figsize=(min(16, 1.2 + 0.62 * len(layers)), 0.62 * n + 1.2))
    ax.set_xlim(-0.9, len(layers) + 0.3)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")
    for q in range(n):
        ax.plot([-0.4, len(layers) + 0.1], [n - 1 - q] * 2, color=GRAY, linewidth=1, zorder=0)
        ax.text(-0.55, n - 1 - q, f"q{q}", ha="right", va="center", fontsize=9, color=INK2)
    colors = {"h": AQUA, "rx": ORANGE, "ry": YELLOW, "rz": BLUE, "rzz": BLUE, "x": INK2}
    for t, layer in enumerate(layers):
        for g in layer:
            if g.name == "cx":
                c, tg = g.qubits
                yc, yt = n - 1 - c, n - 1 - tg
                ax.plot([t, t], [yc, yt], color=INK, linewidth=1.2, zorder=1)
                ax.add_patch(Circle((t, yc), 0.08, color=INK, zorder=2))
                ax.add_patch(Circle((t, yt), 0.17, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=2))
                ax.plot([t - 0.17, t + 0.17], [yt, yt], color=INK, linewidth=1.2, zorder=3)
                ax.plot([t, t], [yt - 0.17, yt + 0.17], color=INK, linewidth=1.2, zorder=3)
                continue
            if g.name == "rzz":
                ys = [n - 1 - q for q in g.qubits]
                ax.plot([t, t], ys, color=BLUE, linewidth=2, zorder=1)
                for yy in ys:
                    ax.add_patch(FancyBboxPatch((t - 0.3, yy - 0.22), 0.6, 0.44, boxstyle="round,pad=0.02",
                                                facecolor=BLUE, edgecolor="none", zorder=2))
                    ax.text(t, yy, "ZZ", ha="center", va="center", fontsize=7, color="white", zorder=3)
                continue
            yy = n - 1 - g.qubits[0]
            col = colors.get(g.name, GRAY)
            ax.add_patch(FancyBboxPatch((t - 0.3, yy - 0.22), 0.6, 0.44, boxstyle="round,pad=0.02",
                                        facecolor=col, edgecolor="none", zorder=2))
            label = g.name.upper() if g.param is None else f"{g.name.upper()}\n{g.param:.2f}"
            ax.text(t, yy, label, ha="center", va="center", fontsize=6.2, color=INK if col == YELLOW else "white",
                    zorder=3, linespacing=0.9)
    total = len(circuit.layers())
    ax.text(-0.4, -0.7, f"{n} qubits · depth {total}" + (f" (first {max_layers} steps shown)" if total > max_layers else ""),
            fontsize=8.5, color=MUTED)
    ax.set_title(title or "Quantum circuit", pad=6)
    return fig


def plot_bloch(vectors, labels=None, title: str | None = None, cols: int = 5):
    """One Bloch sphere per qubit. Arrow length < 1 means the qubit is entangled."""
    plt = _plt()
    vectors = np.asarray(vectors)
    n = len(vectors)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(2.6 * min(cols, n), 2.8 * rows))
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    for i, vec in enumerate(vectors):
        ax = fig.add_subplot(rows, min(cols, n), i + 1, projection="3d")
        ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRID, linewidth=0.4)
        for a, b, c in [((-1, 1), (0, 0), (0, 0)), ((0, 0), (-1, 1), (0, 0)), ((0, 0), (0, 0), (-1, 1))]:
            ax.plot(a, b, c, color=GRAY, linewidth=0.6)
        ax.quiver(0, 0, 0, *vec, color=BLUE, linewidth=2.2, arrow_length_ratio=0.15)
        ax.text(0, 0, 1.25, "|0⟩", fontsize=8, color=INK2, ha="center")
        ax.text(0, 0, -1.45, "|1⟩", fontsize=8, color=INK2, ha="center")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.axis("off")
        name = labels[i] if labels else f"qubit {i}"
        ax.set_title(f"{name}\nP(1) = {(1 - vec[2]) / 2:.2f} · |r| = {np.linalg.norm(vec):.2f}",
                     fontsize=8.5, fontweight="normal", loc="center")
    fig.suptitle(title or "Each qubit on the Bloch sphere (|r| < 1 → entangled with the others)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Problem pictures
# ---------------------------------------------------------------------------
def plot_route(names, coords, tour, km: float | None = None, title: str | None = None):
    plt = _plt()
    lats = [coords[i][0] for i in tour]
    lons = [coords[i][1] for i in tour]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(lons, lats, color=BLUE, linewidth=2, zorder=1, solid_capstyle="round")
    ax.scatter(lons[1:-1], lats[1:-1], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.scatter([lons[0]], [lats[0]], s=140, marker="s", color=INK, edgecolor=SURFACE, linewidth=2, zorder=3)
    for step, i in enumerate(tour[:-1]):
        lab = names[i] if step == 0 else f"{step}. {names[i]}"
        ax.annotate(lab, (coords[i][1], coords[i][0]), xytext=(7, 6), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _clean(ax, "both")
    ax.set_title(title or ("Best visiting order" + (f": {km:.1f} km" if km else "")))
    return fig


def plot_maxcut(problem, bits, graph=None, title: str | None = None, seed: int = 3):
    plt = _plt()
    import networkx as nx
    g = graph if graph is not None else nx.Graph([(i, j) for i, j, _ in problem.edges])
    pos = nx.kamada_kawai_layout(g)
    x = np.asarray(bits)
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    cut = [(i, j) for i, j, _ in problem.edges if x[i] != x[j]]
    same = [(i, j) for i, j, _ in problem.edges if x[i] == x[j]]
    nx.draw_networkx_edges(g, pos, edgelist=same, edge_color=GRAY, width=1.2, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=cut, edge_color=INK, width=2, style="solid", ax=ax)
    for side, col, lab in [(0, BLUE, "Group A"), (1, ORANGE, "Group B")]:
        nodes = [v for v in g.nodes if x[v] == side]
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=col, node_size=420,
                               edgecolors=SURFACE, linewidths=2, ax=ax, label=lab)
    labels = {v: (problem.names[v] if problem.names else str(v)) for v in g.nodes}
    for v, (px, py) in pos.items():
        ax.text(px, py + 0.075, labels[v], ha="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.axis("off")
    ax.set_title(title or f"Max-Cut: {len(cut)} of {len(problem.edges)} links cross the split (black)")
    return fig


# ---------------------------------------------------------------------------
# QAOA behaviour
# ---------------------------------------------------------------------------
def landscape(qubo, n_grid: int = 40):
    """p=1 QAOA energy <C>(gamma, beta) on a grid (normalised cost)."""
    from .solvers.qaoa import qaoa_state
    E = qubo.all_energies()
    cost = (E - E.mean()) / (E.std() or 1.0)
    gs = np.linspace(0, np.pi, n_grid)
    bs = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    Z = np.empty((n_grid, n_grid))
    for a, b in enumerate(bs):
        for c, g in enumerate(gs):
            psi = qaoa_state(cost, [g], [b], qubo.n)
            Z[a, c] = float(np.abs(psi) ** 2 @ cost)
    return gs, bs, Z


def plot_landscape(qubo, n_grid: int = 40, title: str | None = None, marker=None):
    plt = _plt()
    gs, bs, Z = landscape(qubo, n_grid)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="Blues_r",
                   extent=[gs[0], gs[-1], bs[0], bs[-1]])
    a, c = np.unravel_index(np.argmin(Z), Z.shape)
    ax.scatter([gs[c]], [bs[a]], s=90, marker="*", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.annotate(f"best p=1 angles\nγ={gs[c]:.2f}, β={bs[a]:.2f}", (gs[c], bs[a]),
                xytext=(12, -28 if a > len(bs) * 0.7 else 12),
                textcoords="offset points", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    if marker is not None:
        ax.scatter([marker[0]], [marker[1]], s=60, color=INK, zorder=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("expected cost ⟨C⟩ (darker = better)")
    cb.outline.set_visible(False)
    ax.set_xlabel("γ  (cost-layer angle)")
    ax.set_ylabel("β  (mixer angle)")
    ax.set_title(title or "QAOA energy landscape (p = 1)")
    return fig


def plot_amplification(qubo, result, valid_mask=None, title: str | None = None):
    """Chance of measuring the best answer (and any valid answer): random vs trained QAOA."""
    plt = _plt()
    from .solvers.qaoa import qaoa_state
    E = qubo.all_energies()
    cost = (E - E.mean()) / (E.std() or 1.0)
    info = result.info
    probs = np.abs(qaoa_state(cost, info["gammas"], info["betas"], qubo.n, info.get("thetas"))) ** 2
    N = len(probs)
    opt = E <= E.min() + 1e-9 * max(1.0, abs(E.min()))
    groups = [f"Best answer ({int(opt.sum())} equally good)" if opt.sum() > 1 else "Best answer"]
    rnd, qa = [100 * opt.sum() / N], [100 * probs[opt].sum()]
    if valid_mask is not None:
        groups.append(f"Any valid answer ({int(valid_mask.sum())} of {N:,})")
        rnd.append(100 * valid_mask.sum() / N)
        qa.append(100 * probs[valid_mask].sum())
    fig, ax = plt.subplots(figsize=(8, 1.4 + 1.1 * len(groups)))
    y = np.arange(len(groups))
    ax.barh(y + 0.19, rnd, height=0.34, color=GRAY, label="Random guessing")
    ax.barh(y - 0.19, qa, height=0.34, color=BLUE, label=f"Trained QAOA (p={info['p']})")
    for yi, r, q in zip(y, rnd, qa):
        ax.text(r * 1.15, yi + 0.19, f"{r:.4f}%", va="center", fontsize=8.5, color=INK2)
        ax.text(q * 1.15, yi - 0.19, f"{q:.2f}%  ({q / r:.0f}×)", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(rnd) / 2, max(qa) * 25)
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("chance of measuring it in one shot (%, log scale)")
    ax.legend(frameon=False, loc="lower right")
    _clean(ax, "x")
    ax.set_title(title or "QAOA amplifies good answers")
    return fig


def plot_devices(rows, title: str | None = None):
    """rows: list of dicts with device, fidelity, p_opt (from benchmark_devices)."""
    plt = _plt()
    names = [r["device"] for r in rows]
    vals = [100 * r["p_opt"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.62 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.5, color=BLUE)
    for yi, r, v in zip(y, rows, vals):
        f = r["fidelity"]
        ftxt = f"≈ {f:.3f}" if f >= 0.01 else "< 0.01 (output ≈ random)"
        ax.text(v + max(vals) * 0.015, yi, f"{v:.2f}%   circuit fidelity {ftxt}",
                va="center", fontsize=8.5, color=INK)
    xlabel = "chance one shot returns the best answer (%)"
    if rows and "random_guess" in rows[0]:
        rg = 100 * rows[0]["random_guess"]
        ax.axvline(rg, color=INK2, linestyle=":", linewidth=1)
        xlabel += f"   ·   dotted line = random guessing ({rg:.2f}%)"
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.6 + 1e-9)
    ax.set_xlabel(xlabel)
    _clean(ax, "x")
    ax.set_title(title or "Same circuit, different quantum hardware (illustrative profiles)")
    return fig


def plot_scaling(markers: dict | None = None, title: str | None = None):
    """Why simulating qubits is hard: statevector memory doubles with every qubit."""
    plt = _plt()
    n = np.arange(1, 51)
    mem = 16 * 2.0 ** n  # complex128 bytes
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(n, mem, color=BLUE, linewidth=2)
    for b, lab in [(16e9, "laptop (16 GB)"), (2e12, "large server (2 TB)"), (1e15, "biggest supercomputers (~1 PB)")]:
        ax.axhline(b, color=GRAY, linewidth=1)
        ax.text(1.2, b * 1.6, lab, fontsize=8.5, color=INK2)
        q = int(np.floor(np.log2(b / 16)))
        ax.scatter([q], [16 * 2.0 ** q], s=40, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(q + 0.6, 16 * 2.0 ** q / 4, f"{q} qubits", fontsize=8.5, color=INK)
    for i, (name, q) in enumerate(sorted((markers or {}).items(), key=lambda t: t[1])):
        ax.scatter([q], [16 * 2.0 ** q], s=45, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
        dx, dy, ha = ((-8, 8, "right") if i % 2 == 0 else (9, -13, "left"))
        ax.annotate(f"{name} ({q} qubits)", (q, 16 * 2.0 ** q), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=ha)
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("memory for the full quantum state (bytes)")
    _clean(ax, "y")
    ax.set_title(title or "Simulating n qubits needs 2ⁿ numbers: the reason real quantum hardware matters")
    return fig


def plot_warm_start(rows, title: str | None = None):
    """rows: list of dicts with label, p_opt, random_guess."""
    plt = _plt()
    y = np.arange(len(rows))
    vals = [100 * r["p_opt"] for r in rows]
    cols = [GRAY if r.get("baseline") else (BLUE if r.get("standard") else ORANGE) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 0.8 + 0.62 * len(rows)))
    ax.barh(y, vals, height=0.5, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("chance one shot returns a best route (%)")
    _clean(ax, "x")
    ax.set_title(title or "Warm-start QAOA locks onto the answer it starts from")
    return fig


# ---------------------------------------------------------------------------
# Quantum-AI
# ---------------------------------------------------------------------------
def plot_qml(res: dict, title: str | None = None):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    order = np.argsort(res["y_train"], kind="stable")
    K = res["kernel_train"][np.ix_(order, order)]
    im = a1.imshow(K, cmap="Blues", vmin=0, vmax=1)
    split = int((res["y_train"] == 0).sum())
    a1.axhline(split - 0.5, color=ORANGE, linewidth=1.2)
    a1.axvline(split - 0.5, color=ORANGE, linewidth=1.2)
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"Quantum kernel matrix ({res['n_qubits']} qubits)\nsorted by class: blocks = similar samples",
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=a1, shrink=0.8)
    cb.set_label("|⟨φ(x)|φ(x′)⟩|²")
    cb.outline.set_visible(False)
    names = list(res["results"])
    accs = [100 * res["results"][k]["accuracy"] for k in names]
    cols = [BLUE if k.startswith("Quantum") else (GRAY if k.startswith("Majority") else ORANGE) for k in names]
    y = np.arange(len(names))
    a2.barh(y, accs, height=0.5, color=cols)
    for yi, v in zip(y, accs):
        a2.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=9)
    a2.set_yticks(y, names)
    a2.invert_yaxis()
    a2.set_xlim(0, 110)
    a2.set_xlabel("test accuracy (%)")
    _clean(a2, "x")
    a2.set_title("Accuracy on unseen test data", fontsize=10.5)
    fig.suptitle(title or res["dataset"], x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_angle_transfer(rows, title: str | None = None):
    plt = _plt()
    x = np.arange(len(rows))
    tr = [100 * r["ratio_trained"] for r in rows]
    tf = [100 * r["ratio_transferred"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.18, tr, width=0.34, color=BLUE, label="trained on this graph")
    ax.bar(x + 0.18, tf, width=0.34, color=ORANGE, label="learned angles (no training)")
    for xi, a, b, r in zip(x, tr, tf, rows):
        ax.text(xi, max(a, b) + 0.8, f"saves {r['training_calls_saved']} runs", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"graph {r['instance'] + 1}\n{r['nodes']} nodes, {r['edges']} links" for r in rows])
    lo = min(tr + tf)
    ax.set_ylim(max(0, lo - 15), 106)
    ax.set_yticks([t for t in range(0, 101, 5) if t >= max(0, lo - 15)])
    ax.set_ylabel("expected cut / best cut (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    _clean(ax, "y")
    ax.set_title(title or "AI for quantum: angles learned on small graphs work on new, bigger ones")
    return fig


# ---------------------------------------------------------------------------
# Infographic
# ---------------------------------------------------------------------------
def infographic(stats: dict, title: str = "Q-Optima at a glance"):
    """Poster: headline numbers, the pipeline, and the honest bottom line.

    stats keys: problems, max_qubits, best_amplification, cnots, qml_quantum, qml_classical,
                transfer_gap, superconducting_fidelity, ion_fidelity."""
    plt = _plt()
    fig = plt.figure(figsize=(13, 8.6))
    fig.text(0.03, 0.955, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.03, 0.92, "Hybrid quantum–classical optimisation and quantum-AI, checked against proven answers",
             fontsize=11, color=INK2)
    tiles = [
        (f"{stats['problems']}", "real-world problem types", "routing · portfolio · plots · Max-Cut", BLUE),
        (f"{stats['max_qubits']}", "qubits simulated exactly", "full statevector, gate by gate", BLUE),
        (f"{stats['best_amplification']:,.0f}×", "best-answer amplification", "QAOA vs random guessing", ORANGE),
        (f"{stats['cnots']}", "two-qubit gates (CNOTs)", "in one 9-qubit, p = 2 route circuit", ORANGE),
        (f"{stats['qml_quantum']:.1%}", "quantum-kernel accuracy", f"classical: {stats['qml_classical']:.1%}", AQUA),
        (f"{stats['transfer_gap']:.1f} pts", "cost of skipping training", "with AI-learned angles", AQUA),
    ]
    for i, (big, label, sub, col) in enumerate(tiles):
        x = 0.03 + (i % 3) * 0.32
        y = 0.70 - (i // 3) * 0.17
        ax = fig.add_axes([x, y, 0.3, 0.15])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=col, alpha=0.09, linewidth=0))
        ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, facecolor=col, linewidth=0))
        ax.text(0.06, 0.62, big, fontsize=24, fontweight="bold", color=INK, transform=ax.transAxes, va="center")
        ax.text(0.06, 0.30, label, fontsize=10.5, color=INK, transform=ax.transAxes, va="center")
        ax.text(0.06, 0.12, sub, fontsize=8.5, color=INK2, transform=ax.transAxes, va="center")
    axf = fig.add_axes([0.02, 0.10, 0.96, 0.36])
    pipeline_flowchart(axf)
    sc = stats["superconducting_fidelity"]
    sc_txt = f"≈ {sc:.2f}" if sc >= 0.01 else "below 0.01 (noise wins)"
    fig.text(0.03, 0.055,
             f"Honest bottom line: at these sizes classical methods are faster and as accurate. The 9-qubit route circuit's "
             f"estimated fidelity on typical superconducting hardware is {sc_txt}; on trapped ions ≈ {stats['ion_fidelity']:.2f}.",
             fontsize=9.5, color=INK2)
    fig.text(0.03, 0.03, "No quantum speed-up is claimed. Hardware figures are illustrative order-of-magnitude profiles.",
             fontsize=9.5, color=INK2)
    return fig
