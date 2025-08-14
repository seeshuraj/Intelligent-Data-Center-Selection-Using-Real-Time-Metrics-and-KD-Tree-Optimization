import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

class RouterMetricsGenerator:
    def __init__(self, num_nodes=10000):
        self.num_nodes = num_nodes
        np.random.seed(42)
    
    def generate_metrics(self):
        """Generate synthetic router metrics matching your thesis ranges"""
        data = []
        base_time = datetime.now()
        
        for i in range(self.num_nodes):
            # Create realistic geographic clustering
            region = i % 5  # 5 regions
            base_latency = 5 + region * 15  # Different base latencies per region
            
            data.append({
                'node_id': f'router_{i:06d}',
                'ip_address': f'10.{region}.{(i//255)%255}.{i%255}',
                'latency_ms': base_latency + np.random.uniform(0, 25.0),
                'throughput_mbps': np.random.uniform(10.0, 100.0),
                'total_storage_gb': np.random.choice([32, 64, 128]),
                'free_storage_gb': np.random.uniform(5.0, 50.0),
                'distance_km': (region * 5) + np.random.uniform(0.5, 15.0),
                'region': f'region_{region}',
                'timestamp': (base_time - timedelta(minutes=i%60)).isoformat()
            })
        
        return pd.DataFrame(data)
    
    def save_csv(self, filename="router_metrics.csv"):
        df = self.generate_metrics()
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} records saved to {filename}")
        return filename

# Generate test data
if __name__ == "__main__":
    generator = RouterMetricsGenerator(1000)  # Start with 1K nodes
    generator.save_csv("data/router_metrics_1k.csv")
    
    # Also generate larger datasets for benchmarking
    for size in [5000, 10000]:
        generator = RouterMetricsGenerator(size)
        generator.save_csv(f"data/router_metrics_{size//1000}k.csv")
        print(f"Generated {size} node dataset")
