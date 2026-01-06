import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plot_results
from skyfield.api import load, wgs84
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


DURATION_MINUTES = 95
TIME_STEP_MIN = 1

CONFIGS = {
    # "starlink": {
    #     "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    #     "alt_min": 530,
    #     "alt_max": 580,
    #     "max_nodes": 10000,
    #     "isl_range": 1200,
    #     "color": "pink",
    # },
    "oneweb": {
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
        "alt_min": 1150,
        "alt_max": 1250,
        "max_nodes": 10000,
        "isl_range": 2500,
        "color": "purple",
    },
}


def fetch_satellite_data(config):
    print(f"Fetching TLE data from {config['url']}...")
    satellites = load.tle_file(config["url"], reload=True)
    print(f"Total satellites found: {len(satellites)}")

    ts = load.timescale()
    t_now = ts.now()
    subset = []

    for sat in satellites:
        try:
            geocentric = sat.at(t_now)
            subpoint = wgs84.subpoint(geocentric)
            height_km = subpoint.elevation.km

            if config["alt_min"] < height_km < config["alt_max"]:
                subset.append(sat)
        except Exception as _:
            continue

        if len(subset) >= config["max_nodes"]:
            break

    print(f"Selected {len(subset)} satellites for analysis")
    return subset


def build_temporal_network(satellites):
    ts = load.timescale()
    t0 = ts.now()
    snapshots = []
    print(f"Simulating {DURATION_MINUTES} minutes of orbital dynamics...")

    for minute in range(DURATION_MINUTES):
        t = ts.utc(
            t0.utc_datetime().year,
            t0.utc_datetime().month,
            t0.utc_datetime().day,
            t0.utc_datetime().hour,
            t0.utc_datetime().minute + minute,
        )
        positions = []

        for sat in satellites:
            try:
                geocentric = sat.at(t)
                pos = geocentric.position.km
                positions.append(pos)
            except Exception as _:
                positions.append([99999, 99999, 99999])

        pos_matrix = np.array(positions)
        dist_matrix = squareform(pdist(pos_matrix))
        adj_matrix = (dist_matrix < ISL_RANGE_KM).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        G = nx.from_numpy_array(adj_matrix)
        snapshots.append(G)

        if minute % 10 == 0:
            print(f"Time T={minute}: generated graph with {G.number_of_nodes()} nodes")

    return snapshots


def get_dynamic_threshold(G, percentile=90):
    degrees = [d for n, d in G.degree()]
    k = int(np.percentile(degrees, percentile))
    return k


def compute_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    elif len(G) > 0:
        gc_nodes = max(nx.connected_components(G), key=len)
        G_gc = G.subgraph(gc_nodes)
        return nx.average_shortest_path_length(G_gc)
    return 0


def get_rich_nodes(G, k):
    return {n for n, d in G.degree() if d > k}


def compute_normalised_rich_club(G, k):
    rc = nx.rich_club_coefficient(G, normalized=False)
    phi_real = rc.get(k, 0)

    if phi_real == 0:
        return 0

    degrees = [d for n, d in G.degree()]
    G_rand = nx.configuration_model(degrees, create_using=nx.Graph)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

    rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
    phi_rand = rc_rand.get(k, 0)

    if phi_rand > 0:
        return phi_real / phi_rand
    return 0


def compute_stability(prev_set, cur_set):
    if not prev_set and not cur_set:
        return 1.0

    intersection = len(prev_set.intersection(cur_set))
    union = len(prev_set.union(cur_set))

    if union == 0:
        return 0
    return intersection / union


def calculate_basic_metrics(temporal_graphs):
    history = {
        "path_length": [],
        "clustering": [],
        "assortativity": [],
        "time_steps": [],
        "rho": [],
        "stability": [],
        "avg_degree": [],
    }

    k = get_dynamic_threshold(temporal_graphs[0])
    print(f"Dynamic rich-club threshold calculated: k > {k}")
    prev_rich_set = set()
    print("Calculating topological metrics...")

    for t, G in enumerate(
        tqdm(temporal_graphs, desc="Processing snapshots", unit="step")
    ):
        degrees = [d for n, d in G.degree()]
        avg_k = np.mean(degrees) if degrees else 0
        length = compute_path_length(G)
        c = nx.average_clustering(G)
        r = nx.degree_assortativity_coefficient(G)
        rho = compute_normalised_rich_club(G, k)
        cur_rich_set = get_rich_nodes(G, k)

        if t == 0:
            stab = 1.0
        else:
            stab = compute_stability(prev_rich_set, cur_rich_set)

        history["stability"].append(stab)
        history["rho"].append(rho)
        history["path_length"].append(length)
        history["clustering"].append(c)
        history["assortativity"].append(r)
        history["time_steps"].append(t)
        history["avg_degree"].append(avg_k)

        prev_rich_set = cur_rich_set

    return history


if __name__ == "__main__":
    all_results = {}
    for name, conf in CONFIGS.items():
        print(f"\n--- STARTING ANALYSIS FOR: {name} ---")
        global ISL_RANGE_KM
        ISL_RANGE_KM = conf["isl_range"]

        sats = fetch_satellite_data(conf)
        temporal_graphs = build_temporal_network(sats)
        results = calculate_basic_metrics(temporal_graphs)
        all_results[name] = results

        ts = load.timescale()
        plot_results.save_all_plots(temporal_graphs, results, name, sats, ts.now())

    print("\nGenerating Comparison Plots...")

    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(
            res["time_steps"], res["rho"], label=name, color=CONFIGS[name]["color"]
        )

    plt.axhline(1.0, color="k", linestyle="--", label="Random Baseline")
    plt.title("Comparative Rich-Club: Starlink vs OneWeb")
    plt.ylabel("Normalised Rich-Club Coefficient")
    plt.xlabel("Time (min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_rho.png", dpi=300)

    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(
            res["time_steps"],
            res["stability"],
            label=name,
            color=CONFIGS[name]["color"],
        )

    plt.title("Comparative Stability: Starlink vs OneWeb")
    plt.ylabel("Jaccard Index")
    plt.xlabel("Time (min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_stability.png", dpi=300)

    print("Comparison plots saved!")
