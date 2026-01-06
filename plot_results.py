import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skyfield.api import wgs84


def plot_degree_distribution(G, filename="degree_distribution.png"):
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return

    avg_deg = np.mean(degrees)

    plt.figure(figsize=(8, 5))
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


def plot_rich_club_rho(results, filename="rich_club_rho.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(
        results["time_steps"], results["rho"], color="#d62728", marker="o", markersize=3
    )
    plt.axhline(1.0, color="k", linestyle="--", label="Random Baseline (No Rich Club)")

    plt.title("Rich-Club Existence (Normalised)")
    plt.ylabel("Norm. Rich-Club Coefficient")
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
    plt.ylabel("Jaccard Index")
    plt.xlabel("Time (Minutes)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


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


def export_to_gephi(G, satellites, t_now, filename="gephi.gexf"):
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
    print(f"Done! {filename} saved.")


def plot_interactive_globe_full(G, satellites, t_now, filename="interactive.html"):
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
        title=f"Network Topology (N={len(lats)})",
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


def save_all_plots(temporal_graphs, results, name, sats, t_now):
    print("Generating individual plots...")
    plot_degree_distribution(
        temporal_graphs[0], f"{name}/{name}-degree-distribution.png"
    )
    plot_path_length(results, f"{name}/{name}-path-length.png")
    plot_clustering(results, f"{name}/{name}-clustering.png")
    plot_assortativity(results, f"{name}/{name}-assortativity.png")
    plot_rich_club_rho(results, f"{name}/{name}-rich-club-rho.png")
    plot_stability(results, f"{name}/{name}-stability.png")
    plot_avg_degree(results, f"{name}/{name}-avg-degree.png")
    plot_rich_club_curve(temporal_graphs[0], f"{name}/{name}-rich-club-curve.png")
    export_to_gephi(temporal_graphs[0], sats, t_now, f"{name}/{name}-gephi.gexf")
    plot_interactive_globe_full(
        temporal_graphs[0], sats, t_now, f"{name}/{name}-interactive.html"
    )
