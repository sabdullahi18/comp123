import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from scipy.spatial.distance import pdist, squareform

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
MAX_NODES = 500
ISL_RANGE_KM = 2000
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

            if 530 < height_km < 580:
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
        valid_indices = []

        for i, sat in enumerate(satellites):
            try:
                geocentric = sat.at(t)
                pos = geocentric.position.km
                positions.append(pos)
                valid_indices.append(i)
            except Exception as _:
                continue

        pos_matrix = np.array(positions)
        dist_matrix = squareform(pdist(pos_matrix))
        adj_matrix = (dist_matrix < ISL_RANGE_KM).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        G = nx.from_numpy_array(adj_matrix)
        snapshots.append(G)

        if minute % 10 == 0:
            avg_deg = np.mean([d for n, d in G.degree()])
            print(
                f"Time T={minute}: generated graph with {G.number_of_nodes()} nodes. Avg degree: {avg_deg:.2f}"
            )

    return snapshots


satellites = fetch_satellite_data()
temporal_graphs = build_temporal_network(satellites)
G_start = temporal_graphs[0]
degrees = [d for n, d in G_start.degree()]

plt.figure(figsize=(10, 6))
plt.hist(
    degrees,
    bins=range(min(degrees), max(degrees) + 1, 1),
    color="skyblue",
    edgecolor="black",
)
plt.title(f"Degree distribution at T=0 (range={ISL_RANGE_KM}km)")
plt.xlabel("Degree (k)")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.savefig("graph.png", dpi=300)

print("Graph is ready in graph.png")
