import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skyfield.api import wgs84
from scipy.signal import find_peaks
from scipy.stats import poisson


def plot_degree_distribution(G, name, filename="degree_distribution.png"):
    degrees = [d for n, d in G.degree()]
    avg_deg = np.mean(degrees)
    plt.figure(figsize=(10, 6))

    plt.hist(
        degrees,
        bins=range(min(degrees), max(degrees) + 1),
        density=True,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label=f"{name} Degree Freq",
    )

    x_range = np.arange(min(degrees), max(degrees) + 1)
    y_poisson = poisson.pmf(x_range, avg_deg)
    plt.plot(x_range, y_poisson, "r--", linewidth=2.5, label="Random Graph (Poisson)")

    plt.axvline(
        avg_deg,
        color="navy",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {avg_deg:.2f}",
    )

    plt.title(f"{name}: Degree Distribution vs. Random Null Model")
    plt.xlabel(r"Degree ($k$)")
    plt.ylabel("Probability / Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_path_length(results, name, filename="path_length.png"):
    plt.figure(figsize=(10, 6))
    x = np.array(results["time_steps"])
    y_real = np.array(results["path_length"])
    y_rand = np.array(results.get("path_length_rand", [0] * len(x)))

    plt.plot(
        x,
        y_rand,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="Random Graph (Theoretical Limit)",
        linewidth=1.5,
    )

    plt.plot(x, y_real, color="purple", linewidth=2.5, label=f"{name} (Empirical)")
    mean_val = np.mean(y_real)
    plt.axhline(
        mean_val,
        color="indigo",
        linestyle=":",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    mid_idx = len(x) // 2
    x_pos = x[mid_idx]
    y_r = y_real[mid_idx]
    y_null = y_rand[mid_idx]
    gap = y_r - y_null
    plt.annotate(
        f"Overhead: +{gap:.1f} hops",
        xy=(x_pos, y_null),
        xytext=(x_pos, y_r),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5, shrinkA=0, shrinkB=0),
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, edgecolor="white"),
    )

    plt.title(f"{name}: Average Shortest Path Length")
    plt.ylabel(r"Avg Path Length ($L$)")
    plt.xlabel("Time (min)")
    plt.legend(loc="center right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_clustering(results, name, filename="clustering.png"):
    plt.figure(figsize=(10, 6))
    x = np.array(results["time_steps"])
    y_real = np.array(results["clustering"])
    y_rand = np.array(results.get("clustering_rand", [0] * len(x)))

    plt.plot(
        x,
        y_rand,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="Random Null Model",
        linewidth=1.5,
    )

    plt.plot(x, y_real, color="orange", linewidth=2.5, label=f"{name} (Empirical)")
    slope, intercept = np.polyfit(x, y_real, 1)
    trend_line = slope * x + intercept
    plt.plot(
        x,
        trend_line,
        color="darkred",
        linestyle=":",
        linewidth=2,
        label=f"Linear Trend (Slope: {slope:.2e})",
    )

    mid_idx = len(x) // 2
    gap_size = y_real[mid_idx] - y_rand[mid_idx]

    plt.annotate(
        rf"Small-World Gap ($\Delta C \approx ${gap_size:.2f})",
        xy=(x[mid_idx], (y_real[mid_idx] + y_rand[mid_idx]) / 2),
        xytext=(x[mid_idx] + 10, (y_real[mid_idx] + y_rand[mid_idx]) / 2),
        arrowprops=dict(facecolor="black", arrowstyle="-["),
        fontsize=9,
        color="black",
        ha="left",
        va="center",
    )

    plt.title(f"{name}: Clustering Coefficient")
    plt.ylabel(r"Avg Clustering Coefficient ($C$)")
    plt.xlabel("Time (min)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_assortativity(results, name, filename="assortativity.png"):
    plt.figure(figsize=(10, 6))
    x = np.array(results["time_steps"])
    y_real = np.array(results["assortativity"])
    y_rand = np.array(results.get("assortativity_rand", [0] * len(x)))

    plt.plot(
        x,
        y_rand,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label="Random Null Model",
        linewidth=1.5,
    )

    plt.plot(x, y_real, color="teal", linewidth=2.5, label=f"{name} (Empirical)")
    slope, intercept = np.polyfit(x, y_real, 1)
    trend_line = slope * x + intercept

    plt.plot(
        x,
        trend_line,
        color="darkorange",
        linestyle=":",
        linewidth=2,
        label=f"Linear Trend (Slope: {slope:.2e})",
    )

    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.text(x[0], 0.02, "Neutral Mixing (r=0)", fontsize=8, color="black", alpha=0.6)
    plt.title(f"{name}: Degree Assortativity")
    plt.ylabel(r"Assortativity Coefficient ($r$)")
    plt.xlabel("Time (min)")
    plt.legend(loc="center right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_avg_degree(results, name, filename="avg_degree.png"):
    plt.figure(figsize=(10, 6))
    x = np.array(results["time_steps"])
    y = np.array(results["avg_degree"])
    plt.plot(x, y, color="navy", linewidth=2, label=r"Avg Degree $\langle k \rangle$")

    global_mean = np.mean(y)
    plt.axhline(
        global_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Global Mean: {global_mean:.2f}",
    )

    peaks, _ = find_peaks(y, height=global_mean + 0.5, distance=10)
    plt.plot(x[peaks], y[peaks], "rx")

    for p in peaks:
        plt.annotate(
            "Polar\nRegion",
            xy=(x[p], y[p]),
            xytext=(x[p], y[p] + 0.5),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            ha="center",
            fontsize=9,
        )

    plt.title(f"{name}: Average Degree")
    plt.ylabel(r"Average Degree $\langle k \rangle$")
    plt.xlabel("Time (min)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def plot_rich_club_curve(G, name, filename="rich_club_curve.png"):
    rc = nx.rich_club_coefficient(G, normalized=False)
    degrees = [d for n, d in G.degree()]
    G_rand = nx.expected_degree_graph(degrees, selfloops=False)
    rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)

    rhos = []
    ks = sorted(list(rc.keys()))
    clean_ks = []
    deg_array = np.array(degrees)

    for k in ks:
        if np.sum(deg_array > k) < 5:
            continue

        phi_real = rc[k]
        phi_rand = rc_rand.get(k, 0)

        if phi_rand > 0:
            rhos.append(phi_real / phi_rand)
            clean_ks.append(k)

    plt.figure(figsize=(9, 6))
    plt.axhline(1.0, color="grey", linestyle="--", label="Random Baseline")

    plt.plot(
        clean_ks,
        rhos,
        marker="o",
        markersize=4,
        color="purple",
        linewidth=2,
        label=f"{name} Rich-Club",
    )

    rich_indices = [i for i, r in enumerate(rhos) if r > 1.05]
    if rich_indices:
        start_k = clean_ks[rich_indices[0]]
        end_k = clean_ks[rich_indices[-1]]
        plt.axvspan(
            start_k,
            end_k,
            color="purple",
            alpha=0.1,
            label=f"Rich Region (k > {start_k})",
        )

        max_rho = max(rhos)
        max_k = clean_ks[np.argmax(rhos)]
        plt.annotate(
            f"Peak: {max_rho:.2f}",
            xy=(max_k, max_rho),
            xytext=(max_k, max_rho + 0.2),
            arrowprops=dict(facecolor="black", shrink=0.05),
            ha="center",
        )

    plt.xlabel("Degree Threshold ($k$)")
    plt.ylabel(r"Norm Rich-Club Coefficient $\rho$")
    plt.title(f"{name}: Rich-Club Profile at T=0 (Structural Cutoff at N<5)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
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


def plot_latitude_degree_correlation(lats, degrees, name, filename):
    plt.figure(figsize=(10, 6))
    plt.hexbin(np.abs(lats), degrees, gridsize=30, cmap="inferno", mincnt=1)
    plt.colorbar(label="Number of Satellites")

    if "starlink" in name:
        plt.axvline(53, color="cyan", linestyle="--", label="Orbital Inclination (53°)")
    elif "oneweb" in name:
        plt.axvline(
            87.9, color="cyan", linestyle="--", label="Orbital Inclination (87.9°)"
        )

    plt.title(f"{name}: Correlation between Latitude and Node Degree")
    plt.xlabel("Latitude (Absolute Degrees)")
    plt.ylabel("Node Degree (k)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")


def save_all_plots(temporal_graphs, results, name, sats, t_now):
    print("Generating individual plots...")
    plot_degree_distribution(
        temporal_graphs[0],
        name,
        f"{name.lower()}/{name.lower()}-degree-distribution.png",
    )
    plot_path_length(results, name, f"{name.lower()}/{name.lower()}-path-length.png")
    plot_clustering(results, name, f"{name.lower()}/{name.lower()}-clustering.png")
    plot_assortativity(
        results, name, f"{name.lower()}/{name.lower()}-assortativity.png"
    )
    plot_avg_degree(results, name, f"{name.lower()}/{name.lower()}-avg-degree.png")
    plot_rich_club_curve(
        temporal_graphs[0], name, f"{name.lower()}/{name.lower()}-rich-club-curve.png"
    )
    export_to_gephi(
        temporal_graphs[0], sats, t_now, f"{name.lower()}/{name.lower()}-gephi.gexf"
    )
    plot_interactive_globe_full(
        temporal_graphs[0],
        sats,
        t_now,
        f"{name.lower()}/{name.lower()}-interactive.html",
    )
