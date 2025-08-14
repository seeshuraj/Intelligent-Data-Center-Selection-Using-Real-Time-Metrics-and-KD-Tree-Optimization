# main.py

import pandas as pd
import numpy as np
import datetime

from collect_metrics import get_latency, get_storage_via_ftp
from normalize import normalize
from kdtree_selector import build_kdtree, query_kdtree
from utils import haversine_distance
from benchmark import benchmark_kdtree
from visualize import plot_metrics_bar, plot_normalized_radar, plot_3d_kdtree

# -----------------------
# Configuration
# -----------------------

# User Location (Example: Dublin)
user_lat, user_lon = 53.3498, -6.2603

# Router metadata
routers = [
    {"ip": "192.168.1.2", "lat": 53.3331, "lon": -6.2489},  # Router 1
    {"ip": "192.168.1.3", "lat": 53.3419, "lon": -6.2675},  # Router 2 (simulated)
]

data = []

# -----------------------
# Collect Data
# -----------------------

for router in routers:
    ip = router["ip"]
    distance = haversine_distance(user_lat, user_lon, router["lat"], router["lon"])
    latency = get_latency(ip)
    throughput = 50.0 if ip != "192.168.1.3" else 30.0  # Simulated different speeds
    total_storage, free_storage = get_storage_via_ftp(ip) if ip == "192.168.1.2" else (57.2, 20.0)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data.append({
        "router_ip": ip,
        "latency_ms": latency,
        "throughput_mbps": throughput,
        "total_storage_gb": total_storage,
        "free_storage_gb": free_storage,
        "distance_km": round(distance, 2),
        "timestamp": timestamp
    })

df = pd.DataFrame(data)
print("\nCollected Data:\n", df)

# -----------------------
# Validate Storage Fetch
# -----------------------
if df["free_storage_gb"].isnull().any():
    print("\nError: Could not fetch storage details for one or more routers.")
else:
    # -----------------------
    # Normalization & Scoring
    # -----------------------
    df["norm_storage"] = normalize(df["free_storage_gb"])
    df["norm_distance"] = normalize(df["distance_km"])
    df["score"] = 0.6 * (1 / df["latency_ms"]) + 0.4 * df["throughput_mbps"]
    df["norm_score"] = normalize(df["score"])

    print("\nNormalized Data:\n", df[["router_ip", "norm_storage", "norm_distance", "norm_score"]])

    # -----------------------
    # KD-Tree Build & Query
    # -----------------------
    features = df[["norm_storage", "norm_distance", "norm_score"]].values
    user_query = np.array([[0.5, 0.2, 0.8]])  # Preferences: mid-storage, close, high perf

    # Benchmark build + query performance
    benchmark_results = benchmark_kdtree(features, user_query, export_csv=True)
    print("\n[Performance Benchmark]:", benchmark_results)

    # Actual selection
    tree = build_kdtree(features)
    dist, idx = query_kdtree(tree, user_query)

    print("\nBest Matched Data Center:")
    print(df.iloc[idx])

    # Visualizations
    plot_metrics_bar(df)
    plot_normalized_radar(df)
    plot_3d_kdtree(df)