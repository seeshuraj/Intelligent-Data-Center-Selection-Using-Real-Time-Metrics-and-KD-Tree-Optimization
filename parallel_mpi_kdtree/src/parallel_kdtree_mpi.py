from mpi4py import MPI
import numpy as np
from sklearn.neighbors import KDTree
import time

class ParallelKDTree:
    def __init__(self, data, metric_names=None):
        """
        Initialize parallel KD-Tree.

        Args:
            data: (N, D) array of points
            metric_names: Names of metrics/dimensions
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.data = data
        self.metric_names = metric_names or [f"dim_{i}" for i in range(data.shape[1])]

        self.local_tree = None
        self.comm_time = 0.0
        self.compute_time = 0.0

        # For correct global index reconstruction
        self.local_sizes = None
        self.prefix_offsets = None

    def distribute_data(self):
        """Distribute data evenly across MPI processes, track local sizes and offsets."""
        if self.rank == 0:
            # Split indices as evenly as possible
            split_indices = np.array_split(np.arange(len(self.data)), self.size)
            chunks = [self.data[idxs] for idxs in split_indices]
            local_sizes = [len(c) for c in chunks]
        else:
            chunks = None
            local_sizes = None

        # Scatter data to all processes
        start_time = time.time()
        local_data = self.comm.scatter(chunks, root=0)
        local_sizes = self.comm.bcast(local_sizes, root=0)
        self.comm_time += time.time() - start_time

        # Build prefix offsets for global indexing
        self.local_sizes = local_sizes
        # prefix_offsets[r] = starting global index for rank r
        self.prefix_offsets = np.cumsum([0] + local_sizes[:-1])

        return local_data

    def build_local_tree(self, local_data):
        """Build KD-Tree locally on each process."""
        start_time = time.time()
        self.local_tree = KDTree(local_data, leaf_size=30)
        self.compute_time += time.time() - start_time

        if self.rank == 0:
            print(f"Process {self.rank}: Built local tree with {len(local_data)} points")

    def parallel_query(self, query_points, k=1):
        """
        Perform parallel nearest-neighbor search.

        Args:
            query_points: (Q, D) array of query points
            k: Number of nearest neighbors

        Returns (on rank 0):
            distances: (Q, k) array of distances
            indices: (Q, k) array of global indices

        On non-root ranks, returns (None, None).
        """
        # Broadcast query points to all processes
        start_time = time.time()
        query_points = self.comm.bcast(query_points, root=0)
        self.comm_time += time.time() - start_time

        # Local search
        start_time = time.time()
        local_distances, local_indices = self.local_tree.query(query_points, k=k)
        self.compute_time += time.time() - start_time

        # Gather results to root process
        start_time = time.time()
        all_distances = self.comm.gather(local_distances, root=0)
        all_indices = self.comm.gather(local_indices, root=0)
        self.comm_time += time.time() - start_time

        if self.rank == 0:
            global_distances = []
            global_indices = []

            # For each query point, merge candidates from all ranks
            for q in range(len(query_points)):
                candidates = []  # (dist, global_idx)

                for r, (dists_r, inds_r) in enumerate(zip(all_distances, all_indices)):
                    offset = self.prefix_offsets[r]
                    for dist, idx in zip(dists_r[q], inds_r[q]):
                        # Map local index on rank r to global index
                        global_idx = offset + idx
                        candidates.append((dist, global_idx))

                # Sort and keep best k
                candidates.sort(key=lambda x: x[0])
                top_k = candidates[:k]
                global_distances.append([d for d, _ in top_k])
                global_indices.append([i for _, i in top_k])

            return np.array(global_distances), np.array(global_indices)

        return None, None

    def profile_communication(self):
        """Return communication and computation profiling information for this rank."""
        total = self.comm_time + self.compute_time
        comm_overhead = (self.comm_time / total * 100.0) if total > 0 else 0.0
        return {
            'rank': self.rank,
            'communication_time': self.comm_time,
            'computation_time': self.compute_time,
            'total_time': total,
            'comm_overhead': comm_overhead,
        }


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generate sample datacenter metrics on rank 0
    if rank == 0:
        np.random.seed(42)
        data = np.random.rand(1000, 4)  # 1000 datacenters, 4 metrics
        metrics = ['latency', 'storage', 'throughput', 'geo_distance']
        query = np.random.rand(1, 4)
        print(f"Starting parallel KD-Tree with {size} processes...")
        print(f"Data shape: {data.shape}, Metrics: {metrics}")
    else:
        data = None
        metrics = None
        query = None

    # Broadcast shared inputs
    data = comm.bcast(data, root=0)
    metrics = comm.bcast(metrics, root=0)
    query = comm.bcast(query, root=0)

    # Build and query parallel KD-Tree
    tree = ParallelKDTree(data, metrics)
    local_data = tree.distribute_data()
    tree.build_local_tree(local_data)

    distances, indices = tree.parallel_query(query, k=5)

    # Gather per-rank profiles and report average communication overhead
    profile = tree.profile_communication()
    all_profiles = comm.gather(profile, root=0)

    if rank == 0:
        print(f"\nNearest neighbors: indices={indices}, distances={distances}")
        avg_comm = sum(p['comm_overhead'] for p in all_profiles) / len(all_profiles)
        print(f"Communication overhead (avg across ranks): {avg_comm:.2f}%")
