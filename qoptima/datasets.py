"""Real-world data sources.

Offline (bundled, always available):
  * ODISHA_CITIES: 15 district towns and landmarks with approximate coordinates
  * real networks shipped with networkx: Florentine families marriage network
    (Padgett & Ansell, 1993) and Zachary's karate club (1977)
  * real measured datasets shipped with scikit-learn: Breast Cancer Wisconsin
    (diagnostic, 569 patients) and Wine (178 samples, chemical analysis)

Online (used when internet is available, e.g. in Colab; graceful fallback otherwise):
  * road_distance_matrix(): real driving distances from the public OSRM server
  * nse_prices(): daily closing prices of NSE-listed stocks via yfinance
"""
from __future__ import annotations

import numpy as np

# Approximate coordinates (lat, lon), good to about 1-2 km. Fine for demos, not survey data.
ODISHA_CITIES: dict[str, tuple[float, float]] = {
    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8830),
    "Phulnakhara": (20.3900, 85.8600),
    "Puri": (19.8135, 85.8312),
    "Konark": (19.8876, 86.0945),
    "Khordha": (20.1824, 85.6167),
    "Jagatsinghpur": (20.2549, 86.1706),
    "Paradip": (20.3165, 86.6114),
    "Kendrapara": (20.5020, 86.4220),
    "Jajpur": (20.8497, 86.3370),
    "Dhenkanal": (20.6587, 85.5981),
    "Nayagarh": (20.1290, 85.0963),
    "Balasore": (21.4934, 86.9135),
    "Berhampur": (19.3150, 84.7941),
    "Sambalpur": (21.4669, 83.9812),
}

NIFTY_SAMPLE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
                "SUNPHARMA.NS", "MARUTI.NS"]


def odisha_places(names: list[str] | int = 6) -> dict[str, tuple[float, float]]:
    """Pick places by name, or the first n (Bhubaneswar first, so it acts as the depot)."""
    if isinstance(names, int):
        names = list(ODISHA_CITIES)[:names]
    return {k: ODISHA_CITIES[k] for k in names}


def haversine_matrix(places: dict[str, tuple[float, float]], road_factor: float = 1.3) -> np.ndarray:
    coords = np.radians(np.array(list(places.values())))
    lat, lon = coords[:, 0][:, None], coords[:, 1][:, None]
    a = np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a)) * road_factor


def road_distance_matrix(places: dict[str, tuple[float, float]], timeout: float = 15.0):
    """Driving distances (km) from the public OSRM demo server, else straight-line x 1.3.

    Returns (matrix, source) with source "osrm" or "haversine". The demo server is
    for light, fair use. For production, run your own OSRM or use a paid API."""
    try:
        import requests
        coords = ";".join(f"{lon},{lat}" for lat, lon in places.values())
        r = requests.get(f"https://router.project-osrm.org/table/v1/driving/{coords}",
                         params={"annotations": "distance"}, timeout=timeout)
        r.raise_for_status()
        d = np.array(r.json()["distances"], dtype=float) / 1000.0
        if d.shape == (len(places), len(places)) and np.isfinite(d).all():
            return d, "osrm"
    except Exception:
        pass
    return haversine_matrix(places), "haversine"


def nse_prices(tickers: list[str] | None = None, period: str = "2y"):
    """Daily closing prices (DataFrame, one column per ticker) from Yahoo Finance.

    Needs `pip install yfinance` and internet. Raises RuntimeError otherwise, so
    callers can fall back to synthetic data and say so."""
    tickers = tickers or NIFTY_SAMPLE[:8]
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is not installed: pip install yfinance") from e
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("no price data returned (offline, or tickers not found)")
    close = data["Close"] if "Close" in data else data
    close = close.dropna(axis=1, how="all").dropna()
    if close.shape[1] < 2:
        raise RuntimeError("fewer than 2 tickers returned data")
    close.columns = [c.replace(".NS", "") for c in close.columns]
    return close


def real_graph(name: str = "florentine"):
    """Real social networks bundled with networkx (nodes relabelled 0..n-1, names kept)."""
    import networkx as nx
    if name == "florentine":
        g = nx.florentine_families_graph()
    elif name == "karate":
        g = nx.karate_club_graph()
    else:
        raise ValueError("choose 'florentine' (15 nodes) or 'karate' (34 nodes)")
    labels = list(g.nodes)
    g = nx.convert_node_labels_to_integers(g, label_attribute="name")
    for e in g.edges:
        g.edges[e].setdefault("weight", 1.0)
    return g, labels


def karate_subgraph(n: int = 12):
    """The n best-connected members of Zachary's karate club (keeps it simulable)."""
    import networkx as nx
    g, _ = real_graph("karate")
    top = sorted(g.degree, key=lambda t: -t[1])[:n]
    sub = g.subgraph([v for v, _ in top]).copy()
    names = [f"member {sub.nodes[v]['name']}" for v in sub.nodes]
    sub = nx.convert_node_labels_to_integers(sub)
    for e in sub.edges:
        sub.edges[e]["weight"] = 1.0
    return sub, names


def classification_dataset(name: str = "breast_cancer"):
    """Real measured data for the quantum-AI demo. Returns (X, y, feature_names, description)."""
    from sklearn import datasets
    if name == "breast_cancer":
        d = datasets.load_breast_cancer()
        return d.data, d.target, list(d.feature_names), \
            "Breast Cancer Wisconsin (diagnostic): 569 tumours, 30 measured features, malignant vs benign"
    if name == "wine":
        d = datasets.load_wine()
        mask = d.target < 2  # two cultivars -> binary problem
        return d.data[mask], d.target[mask], list(d.feature_names), \
            "Wine recognition: chemical analysis of 130 wines from two cultivars"
    raise ValueError("choose 'breast_cancer' or 'wine'")
