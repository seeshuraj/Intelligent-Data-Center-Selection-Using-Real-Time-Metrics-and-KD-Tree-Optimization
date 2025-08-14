# benchmark.py

import time
import pandas as pd
from kdtree_selector import build_kdtree, query_kdtree

def benchmark_kdtree(features, query_vector, export_csv=False, csv_path="benchmark_metrics.csv"):
    """
    Measures build and query time for KD-Tree.

    Args:
        features (np.ndarray): Normalized feature vectors for KDTree.
        query_vector (np.ndarray): Vector to query the KDTree.
        export_csv (bool): Whether to save the benchmarking results to CSV.
        csv_path (str): Path to save CSV.

    Returns:
        dict: A dictionary with build time and query time.
    """
    start_build = time.time()
    tree = build_kdtree(features)
    end_build = time.time()

    start_query = time.time()
    dist, idx = query_kdtree(tree, query_vector)
    end_query = time.time()

    build_time_ms = round((end_build - start_build) * 1000, 2)
    query_time_ms = round((end_query - start_query) * 1000, 2)

    results = {
        "nodes": len(features),
        "build_time_ms": build_time_ms,
        "query_time_ms": query_time_ms
    }

    if export_csv:
        df = pd.DataFrame([results])
        df.to_csv(csv_path, index=False)
        print(f"[✓] Benchmark saved to {csv_path}")

    return results
