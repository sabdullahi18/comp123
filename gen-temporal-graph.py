import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from scipy.spatial.distance import pdist, squareform

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
MAX_NODES = 500
ISL_RANGE_KM = 2000
ALTITUDE_MIN = 530
ALTITUDE_MAX = 580
DURATION_MINUTES = 95
TIME_STEP_MIN = 1


def fetch_satellite_data():
    print("Fetching TLE data from CelesTrak...")
    satellites = load.tle_file(TLE_URL)
    print(f"Total satellites found: {len(satellites)}")

    ts = load.timescale()
    t_now = ts.now()
    subset = []

    for sat in satellites:
        try:
            geocentric = sat.at(t_now)
            subpoint = wgs84.subpoint(geocentric)
            height_km = subpoint.elevation.km

            if ALTITUDE_MIN < height_km < ALTITUDE_MAX:
                subset.append(sat)
        except Exception as _:
            continue

        if len(subset) >= MAX_NODES:
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


def calculate_basic_metrics(temporal_graphs):
    history = {
        "path_length": [],
        "clustering": [],
        "assortativity": [],
        "time_steps": [],
    }

    print("Calculating topological metrics...")

    for t, G in enumerate(temporal_graphs):
        if nx.is_connected(G):
            length = nx.average_shortest_path_length(G)
        else:
            if len(G) > 0:
                gc_nodes = max(nx.connected_components(G), key=len)
                G_gc = G.subgraph(gc_nodes).copy()
                length = nx.average_shortest_path_length(G_gc)

        c = nx.average_clustering(G)
        r = nx.degree_assortativity_coefficient(G)

        history["path_length"].append(length)
        history["clustering"].append(c)
        history["assortativity"].append(r)
        history["time_steps"].append(t)

    return history


def plot_results(results):
    time = results["time_steps"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax1.plot(time, results["path_length"], color="purple", linewidth=2)
    ax1.set_ylabel("Avg Path Length")
    ax1.set_title("Network Efficiency (Latency)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, results["clustering"], color="orange", linewidth=2)
    ax2.set_ylabel("Clustering Coeff")
    ax2.set_title("Local Robustness")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, results["assortativity"], color="teal", linewidth=2)
    ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Assortativity")
    ax3.set_xlabel("Time (Minutes)")
    ax3.set_title("Mixing Pattern")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("basic_metrics.png", dpi=300)
    print("Saved basic_metrics.png")


if __name__ == "__main__":
    sats = fetch_satellite_data()
    temporal_graphs = build_temporal_network(sats)
    results = calculate_basic_metrics(temporal_graphs)
    plot_results(results)
