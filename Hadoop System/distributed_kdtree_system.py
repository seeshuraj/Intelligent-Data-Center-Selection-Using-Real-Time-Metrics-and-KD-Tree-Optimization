import json
import pickle
import base64
import hashlib
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KDTree
from typing import List, Dict, Tuple

class DistributedKDTreeSystem:
    def __init__(self, num_partitions=5):
        self.num_partitions = num_partitions
        self.norm_params = {
            'latency_ms': {'min': 3.0, 'max': 100.0},
            'throughput_mbps': {'min': 10.0, 'max': 100.0},
            'free_storage_gb': {'min': 5.0, 'max': 60.0},
            'distance_km': {'min': 0.5, 'max': 25.0}
        }
        self.partitions = {}
        
    def normalize_metric(self, value: float, metric_type: str) -> float:
        """Normalize a single metric value"""
        params = self.norm_params[metric_type]
        return (value - params['min']) / (params['max'] - params['min'] + 1e-9)
    
    def get_partition_key(self, node_id: str) -> int:
        """Determine partition for a node based on hash"""
        hash_obj = hashlib.md5(node_id.encode())
        return int(hash_obj.hexdigest()[:8], 16) % self.num_partitions
    
    def mapper_phase(self, data_file: str) -> Dict[int, List[Dict]]:
        """Map phase: partition and normalize data"""
        print("Starting Mapper Phase...")
        
        # Load data
        df = pd.read_csv(data_file)
        partitioned_data = {i: [] for i in range(self.num_partitions)}
        
        for _, row in df.iterrows():
            # Normalize features
            normalized_features = [
                self.normalize_metric(row['latency_ms'], 'latency_ms'),
                self.normalize_metric(row['free_storage_gb'], 'free_storage_gb'),
                self.normalize_metric(row['throughput_mbps'], 'throughput_mbps'),
                self.normalize_metric(row['distance_km'], 'distance_km')
            ]
            
            # Create record
            record = {
                'node_id': row['node_id'],
                'ip_address': row['ip_address'],
                'features': normalized_features,
                'raw_metrics': {
                    'latency_ms': row['latency_ms'],
                    'throughput_mbps': row['throughput_mbps'],
                    'free_storage_gb': row['free_storage_gb'],
                    'distance_km': row['distance_km'],
                    'region': row['region'] if 'region' in row else 'unknown'
                }
            }
            
            # Assign to partition
            partition_id = self.get_partition_key(row['node_id'])
            partitioned_data[partition_id].append(record)
        
        print(f"Mapped {len(df)} records to {self.num_partitions} partitions")
        for pid, records in partitioned_data.items():
            print(f"  Partition {pid}: {len(records)} records")
        
        return partitioned_data
    
    def reducer_phase(self, partitioned_data: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """Reducer phase: build KD-Trees for each partition"""
        print("Starting Reducer Phase...")
        
        partition_trees = {}
        
        for partition_id, records in partitioned_data.items():
            if len(records) == 0:
                continue
                
            print(f"Building KD-Tree for partition {partition_id} ({len(records)} nodes)...")
            
            # Extract features and metadata
            features = np.array([record['features'] for record in records])
            metadata = [record for record in records]
            
            # Build KD-Tree
            try:
                kdtree = KDTree(features, leaf_size=30)
                
                partition_trees[partition_id] = {
                    'kdtree': kdtree,
                    'metadata': metadata,
                    'features': features,
                    'node_count': len(metadata)
                }
                
                print(f"  ✓ Partition {partition_id}: KD-Tree built successfully")
                
            except Exception as e:
                print(f"  ✗ Error building KD-Tree for partition {partition_id}: {e}")
        
        return partition_trees
    
    def build_distributed_index(self, data_file: str) -> Dict[int, Dict]:
        """Complete MapReduce workflow"""
        start_time = time.time()
        
        # Mapper phase
        partitioned_data = self.mapper_phase(data_file)
        
        # Reducer phase  
        self.partitions = self.reducer_phase(partitioned_data)
        
        build_time = time.time() - start_time
        total_nodes = sum(p['node_count'] for p in self.partitions.values())
        
        print(f"\n�� Build Summary:")
        print(f"   Total nodes: {total_nodes:,}")
        print(f"   Partitions: {len(self.partitions)}")
        print(f"   Build time: {build_time:.2f}s")
        print(f"   Nodes/sec: {total_nodes/build_time:,.0f}")
        
        return self.partitions
    
    def query_distributed_index(self, query_metrics: Dict[str, float], k: int = 1) -> Dict:
        """Query all partitions and return best match"""
        if not self.partitions:
            raise ValueError("No partitions built. Call build_distributed_index first.")
        
        # Normalize query
        query_vector = np.array([
            self.normalize_metric(query_metrics['latency_ms'], 'latency_ms'),
            self.normalize_metric(query_metrics['free_storage_gb'], 'free_storage_gb'),
            self.normalize_metric(query_metrics['throughput_mbps'], 'throughput_mbps'),
            self.normalize_metric(query_metrics['distance_km'], 'distance_km')
        ]).reshape(1, -1)
        
        print(f"🔍 Querying for: {query_metrics}")
        print(f"   Normalized: {query_vector.flatten()}")
        
        candidates = []
        
        # Query each partition
        for partition_id, tree_data in self.partitions.items():
            try:
                kdtree = tree_data['kdtree']
                metadata = tree_data['metadata']
                
                distances, indices = kdtree.query(query_vector, k=min(k, len(metadata)))
                
                for dist, idx in zip(distances[0], indices[0]):
                    candidates.append({
                        'partition_id': partition_id,
                        'distance': float(dist),
                        'node_data': metadata[idx],
                        'score': 1.0 / (1.0 + dist)  # Convert distance to score
                    })
                    
            except Exception as e:
                print(f"Error querying partition {partition_id}: {e}")
        
        # Sort by distance (best first)
        candidates.sort(key=lambda x: x['distance'])
        
        if candidates:
            best = candidates[0]
            print(f"\n�� Best Match Found:")
            print(f"   Partition: {best['partition_id']}")
            print(f"   Node: {best['node_data']['node_id']}")
            print(f"   Distance: {best['distance']:.4f}")
            print(f"   Score: {best['score']:.4f}")
            
            return best
        
        return None
    
    def benchmark_performance(self, data_file: str, num_queries: int = 100):
        """Benchmark the distributed system"""
        print(f"\n🏃 Running Performance Benchmark...")
        
        # Build index
        build_start = time.time()
        self.build_distributed_index(data_file)
        build_time = time.time() - build_start
        
        # Generate random queries
        query_times = []
        successful_queries = 0
        
        for i in range(num_queries):
            # Random query in valid ranges
            query = {
                'latency_ms': np.random.uniform(5, 80),
                'free_storage_gb': np.random.uniform(10, 50),
                'throughput_mbps': np.random.uniform(15, 90),
                'distance_km': np.random.uniform(1, 20)
            }
            
            query_start = time.time()
            result = self.query_distributed_index(query)
            query_time = time.time() - query_start
            
            if result:
                successful_queries += 1
                query_times.append(query_time * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_query_time = np.mean(query_times)
        p95_query_time = np.percentile(query_times, 95)
        
        print(f"\n�� Performance Results:")
        print(f"   Build time: {build_time:.2f}s")
        print(f"   Successful queries: {successful_queries}/{num_queries}")
        print(f"   Avg query time: {avg_query_time:.2f}ms")
        print(f"   95th percentile: {p95_query_time:.2f}ms")
        print(f"   Queries/sec: {1000/avg_query_time:.0f}")

# Example usage and testing
def main():
    print("🚀 Distributed KD-Tree System for Data Center Selection")
    print("=" * 60)
    
    # Initialize system
    system = DistributedKDTreeSystem(num_partitions=5)
    
    # Test with 1K dataset
    data_file = "data/router_metrics_1k.csv"
    
    if not pd.io.common.file_exists(data_file):
        print(f"❌ Data file {data_file} not found!")
        print("Please run data_generator.py first")
        return
    
    # Build distributed index
    system.build_distributed_index(data_file)
    
    # Test queries
    test_queries = [
        {'latency_ms': 15.0, 'free_storage_gb': 40.0, 'throughput_mbps': 60.0, 'distance_km': 5.0},
        {'latency_ms': 8.0, 'free_storage_gb': 50.0, 'throughput_mbps': 80.0, 'distance_km': 2.0},
        {'latency_ms': 25.0, 'free_storage_gb': 20.0, 'throughput_mbps': 30.0, 'distance_km': 15.0}
    ]
    
    print(f"\n🧪 Testing Sample Queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}:")
        result = system.query_distributed_index(query)
    
    # Run benchmark
    system.benchmark_performance(data_file, num_queries=50)

if __name__ == "__main__":
    main()
