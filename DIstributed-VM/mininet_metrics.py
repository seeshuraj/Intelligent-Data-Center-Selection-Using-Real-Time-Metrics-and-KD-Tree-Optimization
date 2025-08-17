#!/usr/bin/env python3

import subprocess
import json
import time
import random
import pandas as pd
from datetime import datetime
import math

class MininetDataCenterCollector:
    """Collect metrics from Mininet-emulated data centers"""
    
    def __init__(self, num_datacenters=50):
        self.num_datacenters = num_datacenters
        self.coordinator_ip = "10.0.0.1"
        self.datacenter_ips = [f"10.0.0.{i+2}" for i in range(num_datacenters)]
        
        # Simulated geographic coordinates for Dublin area
        self.base_lat = 53.3498
        self.base_lon = -6.2603
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def ping_latency(self, target_ip):
        """Measure ping latency to target IP"""
        try:
            cmd = ["ping", "-c", "5", target_ip]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse average latency from ping output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'rtt min/avg/max/mdev' in line:
                        avg_latency = float(line.split('/')[5])
                        return avg_latency
            return None
        except Exception as e:
            print(f"Ping error for {target_ip}: {e}")
            return None
    
    def simulate_storage_capacity(self, dc_id):
        """Simulate storage capacity for data center"""
        # Base storage between 100-2000 GB
        base_storage = random.randint(100, 2000)
        
        # Simulate usage pattern (70-95% used)
        usage_percent = random.uniform(0.7, 0.95)
        used_storage = base_storage * usage_percent
        free_storage = base_storage - used_storage
        
        return base_storage, free_storage
    
    def simulate_throughput(self, dc_id):
        """Simulate network throughput for data center"""
        # Throughput between 10-1000 Mbps
        return random.randint(10, 1000)
    
    def generate_coordinates(self, dc_id):
        """Generate realistic coordinates around Dublin area"""
        # Scatter points within ~50km radius of Dublin
        radius_deg = 0.5  # Approximately 50km
        
        lat = self.base_lat + random.uniform(-radius_deg, radius_deg)
        lon = self.base_lon + random.uniform(-radius_deg, radius_deg)
        
        return lat, lon
    
    def collect_all_metrics(self):
        """Collect metrics from all simulated data centers"""
        metrics_data = []
        
        print(f"Collecting metrics from {self.num_datacenters} data centers...")
        
        for i, dc_ip in enumerate(self.datacenter_ips):
            dc_id = f"dc{i+1}"
            
            print(f"Collecting metrics from {dc_id} ({dc_ip})...")
            
            # Measure real network latency via ping
            latency = self.ping_latency(dc_ip)
            
            # Generate simulated storage data
            total_storage, free_storage = self.simulate_storage_capacity(i+1)
            
            # Generate simulated throughput
            throughput = self.simulate_throughput(i+1)
            
            # Generate coordinates
            lat, lon = self.generate_coordinates(i+1)
            
            # Calculate distance from coordinator
            distance = self.haversine_distance(
                self.base_lat, self.base_lon, lat, lon
            )
            
            # Create metrics record
            metrics_record = {
                'node_id': dc_id,
                'ip_address': dc_ip,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'latency_ms': latency if latency else random.uniform(1, 100),
                'throughput_mbps': throughput,
                'total_storage_gb': total_storage,
                'free_storage_gb': free_storage,
                'latitude': lat,
                'longitude': lon,
                'distance_km': distance,
                'region': 'dublin_area'
            }
            
            metrics_data.append(metrics_record)
        
        return metrics_data
    
    def save_metrics_csv(self, metrics_data, filename='mininet_datacenter_metrics.csv'):
        """Save metrics to CSV file"""
        df = pd.DataFrame(metrics_data)
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
        return filename
    
    def generate_synthetic_time_series(self, num_samples=1000):
        """Generate synthetic time series data for extended testing"""
        all_metrics = []
        
        print(f"Generating {num_samples} synthetic metric samples...")
        
        for sample in range(num_samples):
            timestamp = datetime.now().timestamp() + (sample * 30)  # 30-second intervals
            
            for i, dc_ip in enumerate(self.datacenter_ips):
                dc_id = f"dc{i+1}"
                
                # Add realistic time-based variation
                base_latency = random.uniform(5, 50)
                latency_variation = math.sin(sample * 0.1) * 10  # Periodic variation
                latency = max(1, base_latency + latency_variation + random.uniform(-2, 2))
                
                # Storage slowly decreases over time
                base_free = 500 - (sample * 0.1)  # Slowly fill up
                free_storage = max(50, base_free + random.uniform(-50, 50))
                
                # Throughput with daily patterns
                daily_pattern = 50 + 40 * math.sin((sample * 30) / (24 * 3600) * 2 * math.pi)
                throughput = max(10, daily_pattern + random.uniform(-20, 20))
                
                lat, lon = self.generate_coordinates(i+1)
                distance = self.haversine_distance(
                    self.base_lat, self.base_lon, lat, lon
                )
                
                metrics_record = {
                    'node_id': dc_id,
                    'ip_address': dc_ip,
                    'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'latency_ms': round(latency, 2),
                    'throughput_mbps': round(throughput, 1),
                    'total_storage_gb': 1000,
                    'free_storage_gb': round(free_storage, 1),
                    'latitude': lat,
                    'longitude': lon,
                    'distance_km': round(distance, 2),
                    'region': 'dublin_area'
                }
                
                all_metrics.append(metrics_record)
        
        return all_metrics

def main():
    """Main function to collect metrics"""
    collector = MininetDataCenterCollector(num_datacenters=500)
    
    print("=== Mininet Data Center Metrics Collector ===")
    print("1. Real metrics collection (requires running Mininet topology)")
    print("2. Generate synthetic time series data")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        # Collect real metrics from Mininet
        metrics = collector.collect_all_metrics()
        filename = collector.save_metrics_csv(metrics, 'mininet_real_metrics.csv')
        
    elif choice == "2":
        # Generate synthetic time series
        metrics = collector.generate_synthetic_time_series(num_samples=1000)
        filename = collector.save_metrics_csv(metrics, 'mininet_synthetic_metrics.csv')
        
    else:
        print("Invalid choice")
        return
    
    print(f"Data collection complete! {len(metrics)} records saved.")
    print(f"You can now use this CSV with your KD-Tree system.")

if __name__ == '__main__':
    main()
