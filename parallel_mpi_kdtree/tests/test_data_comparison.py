import sys
import os
import numpy as np
from mpi4py import MPI

# Adjust Python path so src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serial_kdtree import SerialKDTree
from src.parallel_kdtree_mpi import ParallelKDTree
from src.metrics_loader import load_metrics, generate_synthetic_metrics


def test_synthetic_data():
    """Test with synthetic data."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # If too many processes for this tiny dataset, skip the test
    if size > 16:
        if rank == 0:
            print(f"\nSkipping TEST 1: SYNTHETIC DATA for {size} processes (only 100 points).")
        return None

    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST 1: SYNTHETIC DATA")
        print("=" * 60)

        np.random.seed(42)
        data = generate_synthetic_metrics(n_datacenters=100, seed=42)
        query = np.random.rand(1, 4)

        serial_tree = SerialKDTree(data)
        serial_dists, serial_inds = serial_tree.query(query, k=5)

        print(f"Data shape: {data.shape}")
        print(f"Serial result - indices: {serial_inds[0]}")
    else:
        data = None
        query = None
        serial_dists = None
        serial_inds = None

    # Broadcast inputs and serial reference to all ranks
    data = comm.bcast(data, root=0)
    query = comm.bcast(query, root=0)
    serial_dists = comm.bcast(serial_dists, root=0)
    serial_inds = comm.bcast(serial_inds, root=0)

    # Build and query parallel tree
    tree = ParallelKDTree(data)
    local_data = tree.distribute_data()
    tree.build_local_tree(local_data)
    parallel_dists, parallel_inds = tree.parallel_query(query, k=5)

    if rank == 0:
        if np.allclose(serial_dists, parallel_dists, rtol=1e-5):
            print("✅ SYNTHETIC DATA TEST PASSED - Results match!")
            return True
        else:
            print("❌ SYNTHETIC DATA TEST FAILED")
            return False

    return None


def test_realistic_data():
    """Test with realistic data from CSV."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST 2: REALISTIC DATA FROM CSV")
        print("=" * 60)

        try:
            data = load_metrics('data/metrics.csv')
            print(f"Data shape: {data.shape}")

            # Check if data is large enough for this process count
            too_small = len(data) < size * 2
            if too_small:
                print(f"  Data too small ({len(data)} points) for {size} processes")
                print("   (Need at least 2 points per process)")
        except FileNotFoundError:
            print(" ERROR: data/metrics.csv not found!")
            data = None
            too_small = False
    else:
        data = None
        too_small = None

    # All ranks must know whether to skip
    too_small = comm.bcast(too_small, root=0)
    data = comm.bcast(data, root=0)

    if too_small or data is None:
        return None

    if rank == 0:
        query = data[0:1]
        k = min(2, len(data) - 1)

        serial_tree = SerialKDTree(data)
        serial_dists, serial_inds = serial_tree.query(query, k=k)
        print(f"Serial result - indices: {serial_inds[0]}")
    else:
        query = None
        serial_dists = None
        serial_inds = None
        k = None

    # Broadcast remaining inputs
    query = comm.bcast(query, root=0)
    serial_dists = comm.bcast(serial_dists, root=0)
    serial_inds = comm.bcast(serial_inds, root=0)
    k = comm.bcast(k, root=0)

    tree = ParallelKDTree(data)
    local_data = tree.distribute_data()
    tree.build_local_tree(local_data)
    parallel_dists, parallel_inds = tree.parallel_query(query, k=k)

    if rank == 0:
        if np.allclose(serial_dists, parallel_dists, rtol=1e-5):
            print(" REALISTIC DATA TEST PASSED - Results match!")
            return True
        else:
            print(" REALISTIC DATA TEST FAILED")
            return False

    return None


def test_correctness_large_synthetic():
    """Test correctness with larger synthetic dataset."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Decide collectively whether to skip based on P vs N
    if rank == 0:
        # 1000 points; if too many processes, correctness test is not meaningful
        too_many_procs = size > 32  # adjust if you want to allow larger P
    else:
        too_many_procs = None

    too_many_procs = comm.bcast(too_many_procs, root=0)
    if too_many_procs:
        if rank == 0:
            print(f"\nSkipping TEST 3: LARGE SYNTHETIC for {size} processes (1000 points).")
        return None

    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST 3: LARGE SYNTHETIC DATA (1000 points)")
        print("=" * 60)

        np.random.seed(42)
        data = generate_synthetic_metrics(n_datacenters=1000, seed=42)
        query = np.random.rand(3, 4)

        serial_tree = SerialKDTree(data)
        serial_dists, serial_inds = serial_tree.query(query, k=5)
        print(f"Data: {data.shape}, Queries: {query.shape}")
    else:
        data = None
        query = None
        serial_dists = None
        serial_inds = None

    # Broadcast data and serial baseline
    data = comm.bcast(data, root=0)
    query = comm.bcast(query, root=0)
    serial_dists = comm.bcast(serial_dists, root=0)
    serial_inds = comm.bcast(serial_inds, root=0)

    tree = ParallelKDTree(data)
    local_data = tree.distribute_data()
    tree.build_local_tree(local_data)
    parallel_dists, parallel_inds = tree.parallel_query(query, k=5)

    if rank == 0:
        all_match = True
        for q_idx in range(len(query)):
            if not np.allclose(serial_dists[q_idx], parallel_dists[q_idx], rtol=1e-5):
                all_match = False
                break

        if all_match:
            print(" LARGE SYNTHETIC DATA TEST PASSED!")
            profile = tree.profile_communication()
            comm_overhead = profile.get('comm_overhead_pct', profile.get('comm_overhead', 0))
            if isinstance(comm_overhead, (int, float)):
                print(f"   Communication overhead: {comm_overhead:.1f}%")
            return True
        else:
            print(" LARGE SYNTHETIC DATA TEST FAILED")
            return False

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = []
    result1 = test_synthetic_data()
    results.append(("Synthetic Data", result1))

    result2 = test_realistic_data()
    results.append(("Realistic Data", result2))

    result3 = test_correctness_large_synthetic()
    results.append(("Large Synthetic", result3))

    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test_name, result in results:
            if result is True:
                status = " PASS"
            elif result is False:
                status = "❌ FAIL"
            else:
                status = "  SKIP"
            print(f"{test_name:.<40} {status}")
        print("=" * 60)

