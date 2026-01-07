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
    "Starlink": {
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        "alt_min": 530,
        "alt_max": 580,
        "max_nodes": 10000,
        "isl_range": 1200,
        "color": "pink",
    },
    "OneWeb": {
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
                if "starlink" in config["url"] and abs(subpoint.latitude.degrees) > 54:
                    continue
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

    G0 = temporal_graphs[0]
    degrees_0 = [d for n, d in G0.degree()]
    k_thresh = get_dynamic_threshold(G0, degrees_0)
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
        lats,
        degs,
        name,
        f"{name.lower()}/{name.lower()}-latitude-degree-correlation.png",
    )


def print_comprehensive_stats_table(all_results, rich_matrices):
    print("\n" + "=" * 60)
    print("UNIFIED LATEX RESULTS TABLE")
    print("=" * 60)

    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{|l|l|l||c|c|c|c|}")
    print(r"\hline")
    print(
        r"\textbf{Network} & \textbf{Category} & \textbf{Metric} & \textbf{Mean} & \textbf{Min} & \textbf{Max} & \textbf{Rand (Mean)} \\"
    )
    print(r"\hline")
    print(r"\hline")

    for name in all_results:
        res = all_results[name]
        mat = rich_matrices.get(name)

        if mat is not None:
            diffs = np.abs(np.diff(mat, axis=1))
            turnover_per_step = np.sum(diffs, axis=0)
            t_mean, t_min, t_max = (
                np.mean(turnover_per_step),
                np.min(turnover_per_step),
                np.max(turnover_per_step),
            )

            durations = []
            for r in range(mat.shape[0]):
                row = mat[r, :]
                count = 0
                for val in row:
                    if val == 1:
                        count += 1
                    elif count > 0:
                        durations.append(count)
                        count = 0
                if count > 0:
                    durations.append(count)

            if durations:
                res_mean, res_min, res_max = (
                    np.mean(durations),
                    np.min(durations),
                    np.max(durations),
                )
            else:
                res_mean, res_min, res_max = 0, 0, 0
        else:
            t_mean = t_min = t_max = res_mean = res_min = res_max = 0

        rows = [
            ("Topology", "Avg Degree $\\langle k \\rangle$", res["avg_degree"], None),
            ("Topology", "Path Length $L$", res["path_length"], "path_length_rand"),
            ("Topology", "Clustering $C$", res["clustering"], "clustering_rand"),
            (
                "Topology",
                "Assortativity $r$",
                res["assortativity"],
                "assortativity_rand",
            ),
            ("Topology", "Rich-Club $\\phi$", res["rich_club"], "rich_club_rand"),
            ("Stability", "Jaccard Index $J$", res["stability"], None),
            ("Stability", "Kendall's Tau $\\tau$", res["kendall_tau"], None),
            (
                "Stability",
                "Residence (min)",
                (res_mean, res_min, res_max),
                None,
            ),
            (
                "Stability",
                "Turnover (nodes/min)",
                (t_mean, t_min, t_max),
                None,
            ),
        ]

        print(f"\\multirow{{{len(rows)}}}{{*}}{{\\textbf{{{name}}}}} ")

        for i, (cat, label, data, rand_key) in enumerate(rows):
            if isinstance(data, tuple):
                mu, mn, mx = data
            else:
                mu = np.mean(data)
                mn = np.min(data)
                mx = np.max(data)

            if rand_key:
                rand_data = res[rand_key]
                rand_str = f"{np.mean(rand_data):.3f}"
            elif cat == "Stability":
                rand_str = "n/a"
            else:
                rand_str = "-"

            cat_str = f"\\textbf{{{cat}}}" if (i == 0 or rows[i - 1][0] != cat) else ""
            print(
                f" & {cat_str} & {label} & {mu:.3f} & {mn:.3f} & {mx:.3f} & {rand_str} \\\\"
            )
            if i == 4:
                print(r"\cline{2-7}")

        print(r"\hline")

    print(r"\end{tabular}")
    print(
        r"\caption{Comprehensive comparison of Network Topology and Stability Metrics}"
    )
    print(r"\label{tab:full_results}")
    print(r"\end{table*}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    all_results = {}
    rich_matrices = {}
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
        rich_matrices[name] = rich_matrix
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

        perform_latitude_analysis(G_mid, sats, t_mid, name)
        plot_results.save_all_plots(temporal_graphs, results, name, sats, t_now)

    print("\nGenerating Comparison Plots...")
    plt.figure(figsize=(10, 6))
    plt.axhline(
        1.0,
        color="k",
        linestyle="--",
        alpha=0.5,
        label="Random Baseline (No Rich Club)",
    )

    for name, res in all_results.items():
        y_data = np.array(res["rho"])
        mean_val = np.mean(y_data)
        color = CONFIGS[name]["color"]

        plt.plot(
            res["time_steps"],
            y_data,
            label=rf"{name} (Mean $\rho$: {mean_val:.2f})",
            color=color,
            linewidth=2,
        )

        plt.axhline(mean_val, color=color, linestyle=":", alpha=0.7)

    plt.title("Comparative Rich-Club Coefficient: Starlink vs OneWeb")
    plt.ylabel(r"Normalised Rich-Club Coefficient ($\rho$)")
    plt.xlabel("Time (min)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_rho.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))

    for name, res in all_results.items():
        y_data = np.array(res["stability"])
        mean_val = np.mean(y_data)
        color = CONFIGS[name]["color"]

        plt.plot(
            res["time_steps"],
            y_data,
            label=f"{name} (Mean J: {mean_val:.2f})",
            color=color,
            linewidth=2,
        )

        plt.axhline(mean_val, color=color, linestyle=":", alpha=0.7)

    plt.title("Comparative Stability of Rich-Club Membership")
    plt.ylabel("Stability (Jaccard Index)")
    plt.xlabel("Time (min)")
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_stability.png", dpi=300)
    plt.close()

    print("Comparison plots saved with stats!")
    print_comprehensive_stats_table(all_results, rich_matrices)
    print("Table generated!")
