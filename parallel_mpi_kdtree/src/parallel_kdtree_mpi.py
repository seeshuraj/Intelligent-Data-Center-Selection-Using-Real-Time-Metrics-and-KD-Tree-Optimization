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
        # Handle ranks with zero local points
        if len(local_data) > 0:
            self.local_tree = KDTree(local_data, leaf_size=30)
        else:
            self.local_tree = None
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
        # Ensure ndarray and 2D
        if not isinstance(query_points, np.ndarray):
            query_points = np.array(query_points, dtype=np.float32)
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)

        # Broadcast query points to all processes
        start_time = time.time()
        query_points = self.comm.bcast(query_points if self.rank == 0 else None, root=0)
        self.comm_time += time.time() - start_time

        n_queries = query_points.shape[0]

        # Local search: k cannot be greater than local number of points
        n_local = self.local_sizes[self.rank] if self.local_sizes is not None else 0
        local_k = min(k, n_local)

        start_time = time.time()
        if self.local_tree is None or local_k == 0:
            # This rank has no candidates
            local_distances = np.full((n_queries, 0), np.inf, dtype=np.float32)
            local_indices = np.full((n_queries, 0), -1, dtype=np.int32)
        else:
            local_distances, local_indices = self.local_tree.query(query_points, k=local_k)
            # sklearn returns 1D when local_k == 1, force 2D
            if local_k == 1:
                local_distances = local_distances[:, np.newaxis]
                local_indices = local_indices[:, np.newaxis]
        self.compute_time += time.time() - start_time

        # Gather results to root process
        start_time = time.time()
        all_distances = self.comm.gather(local_distances, root=0)
        all_indices = self.comm.gather(local_indices, root=0)
        self.comm_time += time.time() - start_time

        if self.rank != 0:
            return None, None

        # Root: merge candidates from all ranks
        global_distances = []
        global_indices = []

        for q in range(n_queries):
            candidates = []  # (dist, global_idx)

            for r, (dists_r, inds_r) in enumerate(zip(all_distances, all_indices)):
                offset = self.prefix_offsets[r]
                # dists_r[q] may be length < k if that rank had few points
                for dist, idx in zip(dists_r[q], inds_r[q]):
                    if idx < 0:   # skip empty placeholders
                        continue
                    global_idx = offset + idx
                    candidates.append((dist, global_idx))

            # Sort and keep best k
            candidates.sort(key=lambda x: x[0])
            top_k = candidates[:k]
            global_distances.append([d for d, _ in top_k])
            global_indices.append([i for _, i in top_k])

        return np.array(global_distances), np.array(global_indices)

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

