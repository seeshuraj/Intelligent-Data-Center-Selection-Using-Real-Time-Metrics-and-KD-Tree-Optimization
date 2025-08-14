# Intelligent Data Center Selection Using KD-Tree Based Performance and Distance Optimization

**Author:** Seeshuraj Bhoopalan  
**Supervisor:** Kirk M. Soodhalter  
**Institution:** Trinity College Dublin, Department of Mathematics  
**Degree:** MSc in High Performance Computing  
**Academic Year:** 2024-2025

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)

## 🚀 Overview

This repository implements an intelligent, real-time data center selection system using **KD-Tree spatial indexing** and **live network/storage metrics**. The system operates efficiently on both edge hardware and distributed cloud environments, providing optimal data center routing decisions with minimal computational overhead.

**Key Innovation:** Combines real-time network performance metrics (latency, storage, throughput) with spatial algorithms (KD-Tree) for intelligent, data-driven routing decisions in resource-constrained environments.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Local System Usage](#local-system-usage)
- [Distributed System Usage](#distributed-system-usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Technical Implementation](#technical-implementation)
- [Research Context](#research-context)
- [Future Work](#future-work)
- [Citation](#citation)
- [Contact](#contact)

## ✨ Features

### Core Capabilities
- **Real-Time Metric Collection:** Ping latency, FTP storage monitoring, throughput estimation
- **KD-Tree Decision Engine:** Fast nearest-neighbor search in normalized 3D metric space
- **Edge-Ready Architecture:** Lightweight deployment on OpenWRT routers with USB storage
- **Distributed Scalability:** MapReduce-compatible design for cloud-scale implementations

### Performance Characteristics
| Configuration | Build Time | Query Time | Accuracy |
|---------------|------------|------------|----------|
| 100 nodes (Local) | 8.3ms | 0.5ms | 100% |
| 1,000 nodes | 24.7ms | 1.1ms | 100% |
| 10,000 nodes (Distributed) | 129.6ms | 6.3ms | 100% |

### Visualization & Analysis
- **3D Scatter Plots:** KD-Tree feature space visualization
- **Radar Charts:** Normalized metric comparison
- **Performance Benchmarking:** Automated CSV export and analysis
- **Real-time Monitoring:** Live metric collection and logging

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data          │    │    Decision      │    │   Scalability   │
│   Collection    │───▶│    Engine        │───▶│   Layer         │
│   Layer         │    │   (KD-Tree)      │    │   (Hadoop)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    ┌─────────┐              ┌─────────┐              ┌─────────┐
    │ Ping    │              │Normalize│              │MapReduce│
    │ FTP     │              │Build    │              │Partition│
    │ iperf3  │              │Query    │              │Aggregate│
    └─────────┘              └─────────┘              └─────────┘
```

**Key Components:**
1. **Metric Collection:** Lightweight utilities for real-time performance monitoring
2. **Decision Engine:** Min-max normalization + KD-Tree nearest-neighbor search
3. **Scalability Framework:** Hadoop MapReduce for distributed KD-Tree construction

## 📁 Repository Structure

```
├── Local-System/                   # Single-node prototype implementation
│   ├── main.py                    # Main orchestration and CLI interface
│   ├── collect_metrics.py         # Real-time metric collection (ping, FTP)
│   ├── kdtree_selector.py         # KD-Tree construction and querying
│   ├── normalize.py               # Min-max normalization utilities
│   ├── benchmark.py               # Performance benchmarking framework
│   ├── utils.py                   # Distance calculation (Haversine)
│   ├── visualize.py               # 3D plots and visualization tools
│   └── requirements.txt           # Python dependencies
├── Hadoop System/                  # Distributed implementation
│   ├── distributed_kdtree_system.py # MapReduce KD-Tree implementation
│   ├── data_generator.py          # Synthetic dataset generation
│   ├── compare_systems.py         # Local vs Distributed comparison
│   ├── requirements.txt           # Distributed system dependencies
│   └── data/                      # Generated datasets (1K, 5K, 10K nodes)
└── README.md                      # This documentation
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Hardware:** OpenWRT routers with USB storage (for local system)
- **Network:** FTP access to router storage, ping connectivity

### Installation

```bash
# Clone repository
git clone https://github.com/seeshuraj/Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization.git
cd Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization

# Setup virtual environment
python3 -m venv kdtree_env
source kdtree_env/bin/activate  # Linux/Mac
# kdtree_env\Scripts\activate  # Windows

# Install dependencies
pip install -r Local-System/requirements.txt
```

### Quick Test

```bash
# Navigate to local system
cd Local-System

# Run complete workflow
python main.py

# Expected Output:
# ✓ Collected metrics from 2 routers
# ✓ KD-Tree built with normalized features
# ✓ Best match: Router 192.168.1.3 (distance: 0.029)
```

## 🖥️ Local System Usage

### Basic Operation

```bash
cd Local-System

# 1. Collect real-time metrics
python collect_metrics.py

# 2. Run full system with visualization
python main.py

# 3. Benchmark performance
python benchmark.py

# 4. Generate visualizations
python visualize.py
```

### Configuration

Edit router configuration in `main.py`:

```python
# User Location (Example: Dublin)
user_lat, user_lon = 53.3498, -6.2603

# Router metadata  
routers = [
    {"ip": "192.168.1.2", "lat": 53.3331, "lon": -6.2489},  # Router 1
    {"ip": "192.168.1.3", "lat": 53.3419, "lon": -6.2675},  # Router 2
]
```

### Sample Output

```
Collected Data:
     router_ip  latency_ms  throughput_mbps  free_storage_gb  distance_km
0  192.168.1.2       3.598             50.0             30.4         2.01
1  192.168.1.3       5.123             30.0             20.0         1.00

Best Matched Data Center:
• Router IP: 192.168.1.3
• Latency: 5.123 ms  
• Free Storage: 20.0 GB
• Distance: 1.00 km
• KD-Tree Score: 0.029
```

## ☁️ Distributed System Usage

### Setup and Execution

```bash
cd "Hadoop System"

# 1. Generate test datasets
python data_generator.py
# Creates: router_metrics_1k.csv, router_metrics_5k.csv, router_metrics_10k.csv

# 2. Run distributed KD-Tree system
python distributed_kdtree_system.py

# 3. Compare local vs distributed performance
python compare_systems.py
```

### MapReduce Simulation

The distributed system simulates Hadoop MapReduce:

- **Mappers:** Partition nodes by hash, normalize metrics
- **Reducers:** Build local KD-Trees per partition
- **Aggregator:** Global nearest-neighbor search across partitions

### Performance Comparison Results

```
📊 Summary Results:
Nodes     Local Build (s)  Local Query (s)  Dist Build (s)  Dist Query (s)
1000      0.024           0.001            0.045           0.003
5000      0.089           0.001            0.156           0.005  
10000     0.178           0.002            0.267           0.008
```

## 📊 Performance Benchmarks

### Scalability Metrics (from Thesis)

| System Type | Nodes | Build Time | Query Time | Memory Usage |
|-------------|-------|------------|------------|--------------|
| Local KD-Tree | 100 | 8.3ms | 0.5ms | ~2MB |
| Local KD-Tree | 1,000 | 24.7ms | 1.1ms | ~15MB |
| Local KD-Tree | 10,000 | 129.6ms | 6.3ms | ~120MB |
| Distributed | 10,000 | 1.7s (parallel) | ~50ms | Distributed |

### Real Hardware Performance

**Testbed:** OpenWRT routers + 64GB USB storage  
**Metrics:** Ping latency (3-26ms), FTP storage (20-52GB free), Distance (1-2.1km)  
**Selection Time:** <10ms end-to-end including metric collection

## 🔧 Technical Implementation

### Metric Collection

```python
# Latency via ping
def get_latency(ip):
    cmd = ["ping", "-c", "5", ip]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse average RTT from output
    
# Storage via FTP
def get_storage_via_ftp(ip):
    ftp = FTP(ip)
    ftp.login('anonymous', '')
    ftp.cwd('shares/USB_Storage')  
    # Calculate used storage by file crawling
```

### KD-Tree Decision Engine

```python
from scipy.spatial import KDTree

# Normalize metrics to [0,1] range
normalized_features = [
    normalize(latencies),
    normalize(storage_available), 
    normalize(throughput),
    normalize(distances)
]

# Build 4D KD-Tree
tree = KDTree(np.array(normalized_features).T)

# Query for best match
distance, index = tree.query([[user_preference_vector]], k=1)
best_router = routers[index[0]]
```

### MapReduce Extension

```python
# Mapper: Partition and normalize
def mapper_phase(data_file):
    for node in load_data(data_file):
        partition_id = hash(node.id) % num_partitions
        emit(partition_id, normalize_metrics(node))

# Reducer: Build local KD-Trees  
def reducer_phase(partition_data):
    return KDTree(partition_data)
    
# Global aggregation
def query_all_partitions(query_vector):
    results = [tree.query(query_vector) for tree in partition_trees]
    return min(results, key=lambda x: x.distance)
```

## 🎓 Research Context

This implementation supports the MSc thesis **"Intelligent Data Center Selection using KD-Tree Based Performance and Distance Optimization"** submitted to Trinity College Dublin.

### Key Research Contributions

1. **Lightweight Edge Computing:** Demonstrates intelligent routing on consumer-grade hardware
2. **Real-time Decision Framework:** Live metrics + spatial algorithms for adaptive selection  
3. **Scalable Architecture:** Blueprint for distributed KD-Tree via MapReduce
4. **Practical Validation:** Physical testbed with OpenWRT routers and USB storage

### Literature Alignment

The approach bridges gaps identified in cloud service selection literature:
- **vs Traditional:** Static geolocation/DNS routing → Dynamic multi-metric selection
- **vs ML-based:** Black-box models requiring training → Interpretable spatial indexing  
- **vs Complex orchestration:** Heavy monitoring overhead → Minimal ping/FTP utilities

## 🔮 Future Work

### Immediate Enhancements
- **Geo-IP Integration:** Replace hardcoded coordinates with MaxMind/Google APIs
- **Machine Learning:** XGBoost/Random Forest for predictive node selection
- **Security:** TLS for FTP, authentication tokens for API access
- **Web Dashboard:** Streamlit/React interface for real-time monitoring

### Research Directions
- **Federated KD-Trees:** Hierarchical decision-making for edge-cloud architectures
- **Mobile Edge Computing:** Support for drone/vehicle-mounted nodes
- **IoT Integration:** Sensor-based metric collection from edge devices
- **Energy Optimization:** Battery-aware selection for sustainable edge computing

## 📖 Citation

If using this work in research, please cite:

```bibtex
@mastersthesis{bhoopalan2025kdtree,
    title={Intelligent Data Center Selection using KD-Tree Based Performance and Distance Optimization},
    author={Seeshuraj Bhoopalan},
    school={Trinity College Dublin},
    department={Department of Mathematics},
    year={2025},
    degree={MSc in High Performance Computing},
    supervisor={Kirk M. Soodhalter}
}
```

## 📞 Contact

**Seeshuraj Bhoopalan**  
📧 Email: [bhoopals@tcd.ie](mailto:bhoopals@tcd.ie)  
🏛️ Institution: Trinity College Dublin  
🔗 LinkedIn: [Connect for collaboration](https://linkedin.com/in/seeshuraj)

---

## 🏆 Acknowledgments

- **Supervisor:** Kirk M. Soodhalter for guidance and support
- **Institution:** Trinity College Dublin, Department of Mathematics  
---

**⭐ Star this repository if you find it useful for your edge computing or distributed systems research!**

*Last Updated: August 2025*
