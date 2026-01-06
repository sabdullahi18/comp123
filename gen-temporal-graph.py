import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plot_results
from skyfield.api import load, wgs84
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
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


def get_dynamic_threshold(G, degrees, percentile=90):
    k = np.percentile(degrees, percentile)
    return k


def compute_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    elif len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph)
    return 0


def get_rich_nodes(G, k):
    return {n for n, d in G.degree() if d > k}


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
        "rich_club": [],
        "stability": [],
        "avg_degree": [],
        "path_length_rand": [],
        "clustering_rand": [],
        "assortativity_rand": [],
        "rich_club_rand": [],
        "rho": [],
        "kendall_tau": [],
    }

    prev_rich_nodes = set()
    degrees_prev = []
    for t, G in enumerate(
        tqdm(temporal_graphs, desc="Processing snapshots", unit="step")
    ):
        degrees = [d for n, d in G.degree()]

        if t > 0:
            tau, _ = kendalltau(degrees_prev, degrees)
            history["kendall_tau"].append(tau)
        else:
            history["kendall_tau"].append(1.0)
        degrees_prev = degrees

        history["avg_degree"].append(np.mean(degrees) if degrees else 0)
        history["clustering"].append(nx.average_clustering(G))
        history["path_length"].append(compute_path_length(G))
        history["time_steps"].append(t * TIME_STEP_MIN)

        try:
            r = nx.degree_assortativity_coefficient(G)
            history["assortativity"].append(r if not np.isnan(r) else 0)
        except Exception as _:
            history["assortativity"].append(0)

        k_thresh = get_dynamic_threshold(G, degrees)
        rc = nx.rich_club_coefficient(G, normalized=False)
        phi_real = rc.get(int(k_thresh), 0)
        history["rich_club"].append(phi_real)

        G_rand = nx.configuration_model(degrees, create_using=nx.Graph)
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        history["clustering_rand"].append(nx.average_clustering(G_rand))
        history["path_length_rand"].append(compute_path_length(G_rand))

        try:
            r_rand = nx.degree_assortativity_coefficient(G_rand)
            history["assortativity_rand"].append(r_rand if not np.isnan(r_rand) else 0)
        except Exception as _:
            history["assortativity_rand"].append(0)

        rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
        phi_rand = rc_rand.get(int(k_thresh), 0)
        history["rich_club_rand"].append(phi_rand)
        history["rho"].append(phi_real / phi_rand if phi_rand > 0 else 1.0)
        cur_rich_nodes = get_rich_nodes(G, k_thresh)
        history["stability"].append(compute_stability(prev_rich_nodes, cur_rich_nodes))
        prev_rich_nodes = cur_rich_nodes

    return history


def perform_latitude_analysis(G, satellites, t_mid, name):
    print("Calculating latitude vs degree analysis...")
    lats = []
    degs = []
    for i, sat in enumerate(satellites):
        if i in G.nodes:
            geo = wgs84.subpoint(sat.at(t_mid))
            lats.append(geo.latitude.degrees)
            degs.append(G.degree(i))
    plot_results.plot_latitude_degree_correlation(
        lats, degs, name, f"{name}/{name}-latitude-degree-correlation.png"
    )


def perform_richness_heatmap(temporal_graphs, satellites, name):
    print("Generating richness heatmap...")
    num_nodes = len(sats)
    num_steps = len(temporal_graphs)
    rich_matrix = np.zeros((num_nodes, num_steps))

    for t_idx, G in enumerate(temporal_graphs):
        degrees = dict(G.degree())
        if not degrees:
            continue

        deg_values = sorted(degrees.values())
        cutoff = np.percentile(deg_values, 95)

        for node, deg in degrees.items():
            if deg >= cutoff:
                rich_matrix[node, t_idx] = 1

    plot_results.plot_richness_heatmap(
        rich_matrix, name, f"{name}/{name}-richness-heatmap.png"
    )


def print_comprehensive_stats_table(all_results):
    print("\n" + "=" * 50)
    print("LATEX STATS TABLE")
    print("=" * 50)

    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{|l|l|c|c|c||c|}")
    print(r"\hline")
    print(
        r"\textbf{Network} & \textbf{Metric} & \textbf{Mean} & \textbf{Min} & \textbf{Max} & \textbf{Random Equiv. (Mean)} \\"
    )
    print(r"\hline")

    metrics_map = [
        ("Avg Degree $\\langle k \\rangle$", "avg_degree", None),
        ("Path Length $L$", "path_length", "path_length_rand"),
        ("Clustering Coeff $C$", "clustering", "clustering_rand"),
        ("Assortativity $r$", "assortativity", "assortativity_rand"),
        ("Rich-Club $\\phi$", "rich_club", "rich_club_rand"),
    ]

    for name in ["starlink", "oneweb"]:
        if name not in all_results:
            continue

        res = all_results[name]
        print(f"\\multirow{{5}}{{*}}{{\\textbf{{{name}}}}} ")

        for label, key, rand_key in metrics_map:
            data = res[key]
            mu = np.mean(data)
            mn = np.min(data)
            mx = np.max(data)

            if rand_key:
                rand_data = res[rand_key]
                rand_mu = np.mean(rand_data)
                rand_str = f"{rand_mu:.3f}"
            else:
                rand_str = "-"

            print(f" & {label} & {mu:.3f} & {mn:.3f} & {mx:.3f} & {rand_str} \\\\")

        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\caption{Statistical comparison of Network Metrics vs. Random Null Models}")
    print(r"\label{tab:network_stats}")
    print(r"\end{table*}")
    print("=" * 50 + "\n")


def print_churn_statistics(rich_matrix, history, name):
    diffs = np.abs(np.diff(rich_matrix, axis=1))
    total_flips = np.sum(diffs)
    avg_turnover = total_flips / rich_matrix.shape[1]

    durations = []
    for r in range(rich_matrix.shape[0]):
        row = rich_matrix[r, :]
        count = 0
        for val in row:
            if val == 1:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)

    avg_residence = np.mean(durations) if durations else 0

    avg_jaccard = np.mean(history["stability"])
    avg_tau = np.mean(history["kendall_tau"])

    print(f"\n--- STABILITY STATISTICS: {name} ---")
    print(f"Mean Jaccard Index:       {avg_jaccard:.3f}")
    print(f"Mean Kendall's Tau:       {avg_tau:.3f}")
    print(f"Avg Residence Time:       {avg_residence:.2f} mins")
    print(f"Turnover Rate:            {avg_turnover:.2f} nodes/min")
    print("--------------------------------------\n")


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
        t_now = ts.now()

        print("Generating richness matrix...")
        num_nodes = len(sats)
        num_steps = len(temporal_graphs)
        rich_matrix = np.zeros((num_nodes, num_steps))
        mid_idx = len(temporal_graphs) // 2
        G_mid = temporal_graphs[mid_idx]
        minute_offset = results["time_steps"][mid_idx]
        t_mid = ts.now() + (minute_offset / 1440.0)

        for t_idx, G in enumerate(temporal_graphs):
            degrees = dict(G.degree())
            if not degrees:
                continue
            cutoff = np.percentile(list(degrees.values()), 95)
            for node, deg in degrees.items():
                if deg >= cutoff:
                    rich_matrix[node, t_idx] = 1

        plot_results.plot_richness_heatmap(
            rich_matrix, name, f"{name}/{name}-richness-heatmap.png"
        )
        plot_results.plot_richness_barcode(
            rich_matrix, name, f"{name}/{name}-barcode.png"
        )
        plot_results.plot_residence_time_hist(
            rich_matrix, name, f"{name}/{name}-residence-hist.png"
        )
        print_churn_statistics(rich_matrix, results, name)
        perform_latitude_analysis(G_mid, sats, t_mid, name)
        plot_results.save_all_plots(temporal_graphs, results, name, sats, t_now)

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
    print_comprehensive_stats_table(all_results)
    print("Table generated!")
