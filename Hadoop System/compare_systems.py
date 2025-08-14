import numpy as np
import pandas as pd
import time
import random
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import KDTree
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeMetrics:
    """Data class for storing node performance metrics"""
    node_id: str
    latency_ms: float
    throughput_mbps: float
    free_storage_gb: float
    distance_km: float
    timestamp: float

class LocalKDTreeSystem:
    """Local KD-Tree implementation for data center selection"""
    
    def __init__(self):
        self.nodes = []
        self.metadata = []
        self.kdtree = None
        self.normalization_stats = {}
        
    def add_node(self, metrics: NodeMetrics):
        """Add a node to the system"""
        self.nodes.append([
            metrics.latency_ms,
            metrics.free_storage_gb, 
            metrics.throughput_mbps,
            metrics.distance_km
        ])
        self.metadata.append({
            'node_id': metrics.node_id,
            'latency_ms': metrics.latency_ms,
            'throughput_mbps': metrics.throughput_mbps,
            'free_storage_gb': metrics.free_storage_gb,
            'distance_km': metrics.distance_km,
            'timestamp': metrics.timestamp
        })
    
    def _calculate_normalization_stats(self):
        """Calculate min/max for normalization"""
        if not self.nodes:
            return
            
        nodes_array = np.array(self.nodes)
        self.normalization_stats = {
            'latency_ms': {'min': nodes_array[:, 0].min(), 'max': nodes_array[:, 0].max()},
            'free_storage_gb': {'min': nodes_array[:, 1].min(), 'max': nodes_array[:, 1].max()},
            'throughput_mbps': {'min': nodes_array[:, 2].min(), 'max': nodes_array[:, 2].max()},
            'distance_km': {'min': nodes_array[:, 3].min(), 'max': nodes_array[:, 3].max()}
        }
    
    def _normalize(self, value: float, metric_name: str) -> float:
        """Normalize a single value using min-max normalization"""
        stats = self.normalization_stats.get(metric_name)
        if not stats or stats['max'] == stats['min']:
            return 0.0
        return (value - stats['min']) / (stats['max'] - stats['min'])
    
    def build_tree(self):
        """Build the KD-Tree from collected nodes"""
        if len(self.nodes) == 0:
            logger.warning("No nodes to build tree")
            return
            
        start_time = time.time()
        
        # Calculate normalization statistics
        self._calculate_normalization_stats()
        
        # Normalize all nodes
        normalized_nodes = []
        for node in self.nodes:
            normalized_node = [
                self._normalize(node[0], 'latency_ms'),
                self._normalize(node[1], 'free_storage_gb'),  
                self._normalize(node[2], 'throughput_mbps'),
                self._normalize(node[3], 'distance_km')
            ]
            normalized_nodes.append(normalized_node)
        
        # Build KD-Tree
        self.kdtree = KDTree(np.array(normalized_nodes))
        
        build_time = time.time() - start_time
        logger.info(f"Built local KD-Tree with {len(self.nodes)} nodes")
        return build_time
    
    def query(self, query_metrics: Dict, k: int = 1):
        """Query the KD-Tree for nearest neighbors"""
        if self.kdtree is None:
            logger.error("KD-Tree not built yet")
            return None
        
        # Normalize query vector
        query_vector = np.array([
            self._normalize(query_metrics['latency_ms'], 'latency_ms'),
            self._normalize(query_metrics['free_storage_gb'], 'free_storage_gb'),
            self._normalize(query_metrics['throughput_mbps'], 'throughput_mbps'),
            self._normalize(query_metrics['distance_km'], 'distance_km')
        ]).reshape(1, -1)
        
        start_time = time.time()
        distances, indices = self.kdtree.query(query_vector, k=k)
        query_time = time.time() - start_time
        
        # ✅ FIXED: Extract scalar values from 2D arrays
        best_index = int(indices[0][0])
        distance_value = float(distances)
        
        return {
            'distance': distance_value,
            'node_data': self.metadata[best_index],
            'query_time': query_time
        }

class DistributedKDTreeSystem:
    """Simulated distributed KD-Tree system using MapReduce paradigm"""
    
    def __init__(self, num_partitions: int = 3):
        self.num_partitions = num_partitions
        self.partitions = [[] for _ in range(num_partitions)]
        self.partition_trees = []
        self.partition_metadata = [[] for _ in range(num_partitions)]
        self.global_normalization_stats = {}
        
    def add_node(self, metrics: NodeMetrics):
        """Add node to appropriate partition based on hash"""
        partition_id = hash(metrics.node_id) % self.num_partitions
        
        node_data = [
            metrics.latency_ms,
            metrics.free_storage_gb,
            metrics.throughput_mbps,
            metrics.distance_km
        ]
        
        metadata = {
            'node_id': metrics.node_id,
            'latency_ms': metrics.latency_ms,
            'throughput_mbps': metrics.throughput_mbps,
            'free_storage_gb': metrics.free_storage_gb,
            'distance_km': metrics.distance_km,
            'timestamp': metrics.timestamp,
            'partition_id': partition_id
        }
        
        self.partitions[partition_id].append(node_data)
        self.partition_metadata[partition_id].append(metadata)
    
    def _calculate_global_normalization_stats(self):
        """Calculate global normalization statistics across all partitions"""
        all_nodes = []
        for partition in self.partitions:
            all_nodes.extend(partition)
        
        if not all_nodes:
            return
            
        nodes_array = np.array(all_nodes)
        self.global_normalization_stats = {
            'latency_ms': {'min': nodes_array[:, 0].min(), 'max': nodes_array[:, 0].max()},
            'free_storage_gb': {'min': nodes_array[:, 1].min(), 'max': nodes_array[:, 1].max()},
            'throughput_mbps': {'min': nodes_array[:, 2].min(), 'max': nodes_array[:, 2].max()},
            'distance_km': {'min': nodes_array[:, 3].min(), 'max': nodes_array[:, 3].max()}
        }
    
    def _normalize(self, value: float, metric_name: str) -> float:
        """Normalize using global statistics"""
        stats = self.global_normalization_stats.get(metric_name)
        if not stats or stats['max'] == stats['min']:
            return 0.0
        return (value - stats['min']) / (stats['max'] - stats['min'])
    
    def build_tree(self):
        """Build KD-Trees for each partition (MapReduce simulation)"""
        start_time = time.time()
        
        # Calculate global normalization stats
        self._calculate_global_normalization_stats()
        
        self.partition_trees = []
        
        for i, partition in enumerate(self.partitions):
            if len(partition) == 0:
                self.partition_trees.append(None)
                continue
                
            # Normalize partition data
            normalized_partition = []
            for node in partition:
                normalized_node = [
                    self._normalize(node[0], 'latency_ms'),
                    self._normalize(node[1], 'free_storage_gb'),
                    self._normalize(node[2], 'throughput_mbps'),
                    self._normalize(node[3], 'distance_km')
                ]
                normalized_partition.append(normalized_node)
            
            # Build partition KD-Tree
            partition_tree = KDTree(np.array(normalized_partition))
            self.partition_trees.append(partition_tree)
        
        build_time = time.time() - start_time
        total_nodes = sum(len(p) for p in self.partitions)
        logger.info(f"Built distributed KD-Tree with {total_nodes} nodes across {self.num_partitions} partitions")
        return build_time
    
    def query(self, query_metrics: Dict, k: int = 1):
        """Query all partitions and aggregate results"""
        if not self.partition_trees:
            logger.error("Partition trees not built yet")
            return None
        
        # Normalize query vector using global stats
        query_vector = np.array([
            self._normalize(query_metrics['latency_ms'], 'latency_ms'),
            self._normalize(query_metrics['free_storage_gb'], 'free_storage_gb'),
            self._normalize(query_metrics['throughput_mbps'], 'throughput_mbps'),
            self._normalize(query_metrics['distance_km'], 'distance_km')
        ]).reshape(1, -1)
        
        start_time = time.time()
        
        best_distance = float('inf')
        best_node = None
        
        # Query each partition
        for partition_id, tree in enumerate(self.partition_trees):
            if tree is None:
                continue
                
            distances, indices = tree.query(query_vector, k=1)
            
            # ✅ FIXED: Extract scalar values from 2D arrays
            distance_value = float(distances[0][0])
            index_value = int(indices)
            
            if distance_value < best_distance:
                best_distance = distance_value
                best_node = self.partition_metadata[partition_id][index_value]
        
        query_time = time.time() - start_time
        
        return {
            'distance': best_distance,
            'node_data': best_node,
            'query_time': query_time
        }

def generate_synthetic_nodes(num_nodes: int) -> List[NodeMetrics]:
    """Generate synthetic node data for testing"""
    nodes = []
    
    for i in range(num_nodes):
        # Generate realistic ranges based on thesis data
        latency = random.uniform(3.0, 100.0)  # 3-100ms
        throughput = random.uniform(10.0, 100.0)  # 10-100 Mbps
        storage = random.uniform(5.0, 64.0)  # 5-64 GB
        distance = random.uniform(0.5, 20.0)  # 0.5-20 km
        
        node = NodeMetrics(
            node_id=f"node_{i:04d}",
            latency_ms=latency,
            throughput_mbps=throughput,
            free_storage_gb=storage,
            distance_km=distance,
            timestamp=time.time()
        )
        nodes.append(node)
    
    return nodes

def run_performance_comparison(node_counts: List[int], num_queries: int = 100):
    """Run performance comparison between local and distributed systems"""
    results = []
    
    for num_nodes in node_counts:
        print(f"\n📊 Testing {num_nodes:,} nodes:")
        print("-" * 30)
        
        # Generate test data
        nodes = generate_synthetic_nodes(num_nodes)
        
        # Test Local KD-Tree
        print("Local KD-Tree:")
        local_system = LocalKDTreeSystem()
        for node in nodes:
            local_system.add_node(node)
        
        local_build_time = local_system.build_tree()
        
        # Test queries
        local_query_times = []
        for _ in range(num_queries):
            query = {
                'latency_ms': random.uniform(5, 50),
                'free_storage_gb': random.uniform(10, 50),
                'throughput_mbps': random.uniform(20, 80),
                'distance_km': random.uniform(1, 10)
            }
            
            result = local_system.query(query)
            if result:
                local_query_times.append(result['query_time'])
        
        local_avg_query = np.mean(local_query_times) if local_query_times else 0
        print(f"  Build time: {local_build_time:.3f}s")
        print(f"  Avg query time: {local_avg_query*1000:.1f}ms")
        print(f"  Sample result: {local_system.query({'latency_ms': 10, 'free_storage_gb': 30, 'throughput_mbps': 50, 'distance_km': 2})['node_data']['node_id']}")
        
        # Test Distributed KD-Tree
        print("\nDistributed KD-Tree:")
        distributed_system = DistributedKDTreeSystem(num_partitions=max(3, num_nodes//1000))
        for node in nodes:
            distributed_system.add_node(node)
        
        distributed_build_time = distributed_system.build_tree()
        
        # Test queries
        distributed_query_times = []
        for _ in range(num_queries):
            query = {
                'latency_ms': random.uniform(5, 50),
                'free_storage_gb': random.uniform(10, 50),
                'throughput_mbps': random.uniform(20, 80),
                'distance_km': random.uniform(1, 10)
            }
            
            result = distributed_system.query(query)
            if result:
                distributed_query_times.append(result['query_time'])
        
        distributed_avg_query = np.mean(distributed_query_times) if distributed_query_times else 0
        print(f"  Build time: {distributed_build_time:.3f}s")
        print(f"  Avg query time: {distributed_avg_query*1000:.1f}ms")
        print(f"  Sample result: {distributed_system.query({'latency_ms': 10, 'free_storage_gb': 30, 'throughput_mbps': 50, 'distance_km': 2})['node_data']['node_id']}")
        
        # Performance comparison
        print(f"\n📈 Performance Comparison:")
        build_speedup = local_build_time / distributed_build_time if distributed_build_time > 0 else 0
        query_overhead = distributed_avg_query / local_avg_query if local_avg_query > 0 else 0
        
        print(f"  Build speedup: {build_speedup:.2f}x {'(local faster)' if build_speedup > 1 else '(distributed faster)'}")
        print(f"  Query overhead: {query_overhead:.1f}x {'(distributed slower)' if query_overhead > 1 else '(distributed faster)'}")
        
        results.append({
            'nodes': num_nodes,
            'local_build_time': local_build_time,
            'local_query_time': local_avg_query,
            'distributed_build_time': distributed_build_time,
            'distributed_query_time': distributed_avg_query,
            'build_speedup': build_speedup,
            'query_overhead': query_overhead
        })
    
    return results

def compare_systems():
    """Main comparison function"""
    print("🔬 Comparing Local vs Distributed KD-Tree Systems")
    print("=" * 60)
    
    # Test with different node counts
    node_counts = [1000, 5000, 10000]
    
    try:
        results = run_performance_comparison(node_counts, num_queries=50)
        
        # Create summary table
        print(f"\n📋 Summary Results:")
        print("-" * 80)
        df = pd.DataFrame(results)
        df_display = df[['nodes', 'local_build_time', 'local_query_time', 'distributed_build_time', 'distributed_query_time']].copy()
        df_display.columns = ['Nodes', 'Local Build (s)', 'Local Query (s)', 'Dist Build (s)', 'Dist Query (s)']
        
        # Format for better display
        for col in ['Local Build (s)', 'Local Query (s)', 'Dist Build (s)', 'Dist Query (s)']:
            df_display[col] = df_display[col].map('{:.3f}'.format)
        
        print(df_display.to_string(index=False))
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📝 Results show KD-Tree performance scaling characteristics")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise

if __name__ == "__main__":
    compare_systems()
