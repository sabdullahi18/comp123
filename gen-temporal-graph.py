import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skyfield.api import load, wgs84
from scipy.spatial.distance import pdist, squareform
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


def plot_degree_distribution(G, filename="degree_distribution.png"):
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return

    avg_deg = np.mean(degrees)

    plt.figure(figsize=(8, 6))
    plt.hist(
        degrees,
        bins=range(int(min(degrees)), int(max(degrees)) + 2),
        color="skyblue",
        edgecolor="black",
        rwidth=0.8,
        label="Node Count",
    )
    plt.axvline(
        avg_deg,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Avg Degree: {avg_deg:.2f}",
    )

    plt.title(f"Degree Distribution (N={len(G.nodes)})")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_path_length(results, filename="path_length.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(results["time_steps"], results["path_length"], color="purple", linewidth=2)
    plt.title("Avg Shortest Path")
    plt.ylabel("Avg Path Length")
    plt.xlabel("Time (Minutes)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_clustering(results, filename="clustering.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(results["time_steps"], results["clustering"], color="orange", linewidth=2)
    plt.title("Avg Clustering Coefficient")
    plt.ylabel("Clustering Coefficient")
    plt.xlabel("Time (Minutes)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_assortativity(results, filename="assortativity.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(results["time_steps"], results["assortativity"], color="teal", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5, label="Neutral")
    plt.title("Degree Assortativity")
    plt.ylabel("Assortativity")
    plt.xlabel("Time (Minutes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_avg_degree(results, filename="avg_degree.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(results["time_steps"], results["avg_degree"], color="blue", linewidth=2)
    plt.title("Average Degree")
    plt.ylabel("Average Degree")
    plt.xlabel("Time (Minutes)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_rich_club_phi(results, filename="rich_club_rho.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(
        results["time_steps"], results["rho"], color="#d62728", marker="o", markersize=3
    )
    plt.axhline(1.0, color="k", linestyle="--", label="Random Baseline (No Rich Club)")

    plt.title("Rich-Club Existence (Normalised)")
    plt.ylabel(r"Norm. Rich-Club Coefficient")
    plt.xlabel("Time (Minutes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_stability(results, filename="stability.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(
        results["time_steps"],
        results["stability"],
        color="#2ca02c",
        marker="s",
        markersize=3,
    )
    plt.title("Rich-Club Stability")
    plt.ylabel("Jaccard Similarity Index")
    plt.xlabel("Time (Minutes)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def save_all_plots(temporal_graphs, results):
    print("Generating individual plots...")

    plot_degree_distribution(temporal_graphs[0])
    plot_path_length(results)
    plot_clustering(results)
    plot_assortativity(results)
    plot_rich_club_phi(results)
    plot_stability(results)
    plot_avg_degree(results)


def plot_rich_club_curve(G, filename="rich_club_curve.png"):
    rc = nx.rich_club_coefficient(G, normalized=False)
    degrees = [d for n, d in G.degree()]
    G_rand = nx.configuration_model(degrees, create_using=nx.Graph)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)

    rhos = []
    ks = sorted(list(rc.keys()))

    for k in ks:
        phi_real = rc[k]
        phi_rand = rc_rand.get(k, 0)
        if phi_rand > 0:
            rhos.append(phi_real / phi_rand)
        else:
            rhos.append(0)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, rhos, marker="o", color="purple")
    plt.axhline(1.0, color="k", linestyle="--", label="Random Baseline")
    plt.xlabel("Degree k")
    plt.ylabel("Normalized Rich-Club Coefficient")
    plt.title("Rich-Club Curve at T=0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def export_to_gephi(G, satellites, t_now, filename="starlink_gephi.gexf"):
    print(f"Exporting {filename} for Gephi...")
    G_export = G.copy()

    for node_id in G_export.nodes():
        try:
            idx = int(node_id)
            if 0 <= idx < len(satellites):
                sat = satellites[idx]
                geocentric = sat.at(t_now)
                subpoint = wgs84.subpoint(geocentric)

                G_export.nodes[node_id]["lat"] = subpoint.latitude.degrees
                G_export.nodes[node_id]["lng"] = subpoint.longitude.degrees
                G_export.nodes[node_id]["altitude"] = subpoint.elevation.km

        except (ValueError, IndexError):
            continue

    nx.write_gexf(G_export, filename)
    print("Done! Gephi file saved.")


def plot_interactive_globe_full(G, satellites, t_now, filename):
    print(f"Generating 3D Network Globe: {filename}...")
    node_positions = {}
    node_degrees = []
    lats, lons, texts = [], [], []

    for node_id in G.nodes():
        try:
            idx = int(node_id)
            if 0 <= idx < len(satellites):
                sat = satellites[idx]
                geocentric = sat.at(t_now)
                subpoint = wgs84.subpoint(geocentric)

                lat = subpoint.latitude.degrees
                lon = subpoint.longitude.degrees
                deg = G.degree(node_id)

                node_positions[node_id] = (lat, lon)

                lats.append(lat)
                lons.append(lon)
                texts.append(f"{sat.name}<br>Degree: {deg}")
                node_degrees.append(deg)
        except Exception as _:
            continue

    edge_lats = []
    edge_lons = []

    for u, v in G.edges():
        if u in node_positions and v in node_positions:
            lat1, lon1 = node_positions[u]
            lat2, lon2 = node_positions[v]

            edge_lats.extend([lat1, lat2, None])
            edge_lons.extend([lon1, lon2, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=edge_lons,
            lat=edge_lats,
            mode="lines",
            line=dict(width=0.5, color="rgba(200, 200, 200, 0.3)"),
            name="ISL Links",
            hoverinfo="none",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            text=texts,
            mode="markers",
            name="Satellites",
            marker=dict(
                size=4,
                color=node_degrees,
                colorscale="Plasma",
                colorbar=dict(title="Degree (k)"),
                cmin=min(node_degrees),
                cmax=max(node_degrees),
                opacity=0.9,
            ),
        )
    )

    fig.update_layout(
        title=f"Starlink Network Topology (N={len(lats)})",
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="rgb(20, 20, 20)",
            showocean=True,
            oceancolor="rgb(10, 10, 30)",
            showcountries=True,
            countrycolor="rgb(50, 50, 50)",
            bgcolor="rgb(0, 0, 0)",
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor="black",
        font=dict(color="white"),
    )

    fig.write_html(filename)
    print(f"Saved! Open '{filename}' in your web browser to interact.")


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
        export_to_gephi(
            temporal_graphs[0], sats, ts.now(), filename=f"{name}-gephi.gexf"
        )
        plot_interactive_globe_full(
            temporal_graphs[0], sats, ts.now(), filename=f"{name}-interactive.html"
        )
    print("\nGenerating Comparison Plots...")

    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(
            res["time_steps"], res["rho"], label=name, color=CONFIGS[name]["color"]
        )

    plt.axhline(1.0, color="k", linestyle="--", label="Random Baseline")
    plt.title("Comparative Rich-Club: Starlink vs OneWeb")
    plt.ylabel(r"Normalised Rich-Club Coefficient ($\rho$)")
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
    plt.ylabel("Jaccard Stability Index")
    plt.xlabel("Time (min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_stability.png", dpi=300)

    print("Comparison plots saved!")
