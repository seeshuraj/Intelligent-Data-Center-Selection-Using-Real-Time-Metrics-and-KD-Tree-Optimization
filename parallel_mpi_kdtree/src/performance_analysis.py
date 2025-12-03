from mpi4py import MPI
import numpy as np
import time
from parallel_kdtree_mpi import ParallelKDTree
from serial_kdtree import SerialKDTree
import json
import os


def _load_existing_results(path: str):
    """Load existing benchmark results (if any) from JSON."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # If file is not a list for some reason, ignore it
        return []
    except Exception:
        # Corrupt or unreadable file → start fresh
        return []


def _save_merged_results(new_results, path: str = "results/benchmark_results.json"):
    """Merge new results with existing ones and save back to JSON (rank 0 only)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing = _load_existing_results(path)

    # Combine existing + new
    combined = existing + new_results

    # De‑duplicate by (data_size, processes)
    seen = set()
    deduped = []
    for r in combined:
        key = (r.get("data_size"), r.get("processes"))
        if key in seen:
            # skip older duplicates
            continue
        seen.add(key)
        deduped.append(r)

    with open(path, "w") as f:
        json.dump(deduped, f, indent=2)


def benchmark():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test parameters
    data_sizes = [500, 1000, 5000]
    dimensions = 4
    results = []

    for data_size in data_sizes:
        if rank == 0:
            data = np.random.rand(data_size, dimensions)
            queries = np.random.rand(10, dimensions)

            # Serial baseline
            serial_start = time.time()
            serial_tree = SerialKDTree(data)
            serial_distances, serial_indices = serial_tree.query(queries, k=5)
            serial_time = time.time() - serial_start
        else:
            data = None
            queries = None
            serial_time = None

        # Broadcast to all
        data = comm.bcast(data, root=0)
        queries = comm.bcast(queries, root=0)

        # Parallel execution
        parallel_start = time.time()
        tree = ParallelKDTree(data)
        local_data = tree.distribute_data()
        tree.build_local_tree(local_data)
        parallel_distances, parallel_indices = tree.parallel_query(queries, k=5)
        parallel_time = time.time() - parallel_start

        if rank == 0:
            speedup = serial_time / parallel_time
            efficiency = speedup / size * 100.0
            profile = tree.profile_communication()

            # Make sure we use the correct key name from profile_communication()
            comm_overhead = profile.get("comm_overhead", 0.0)

            result = {
                "data_size": int(data_size),
                "processes": int(size),
                "serial_time": float(serial_time),
                "parallel_time": float(parallel_time),
                "speedup": float(speedup),
                "efficiency": float(efficiency),
                "comm_overhead_percent": float(comm_overhead),
            }
            results.append(result)

            print(f"\n=== Data Size: {data_size} ===")
            print(f"Serial time: {serial_time:.4f}s")
            print(f"Parallel time ({size} processes): {parallel_time:.4f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {efficiency:.2f}%")
            print(f"Communication overhead: {comm_overhead:.2f}%")

    # Only rank 0 writes to disk
    if rank == 0:
        _save_merged_results(results, path="results/benchmark_results.json")
        print("\n✓ Results merged into results/benchmark_results.json")


if __name__ == "__main__":
    benchmark()
