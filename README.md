# Intelligent Data Center Selection Using KD-Tree Based Performance and Distance Optimization

[![MSc Thesis](https://img.shields.io/badge/MSc-High%20Performance%20Computing-orange.svg)](https://www.tcd.ie/)

**Author:** Seeshuraj Bhoopalan  
**Supervisor:** Kirk M. Soodhalter  
**Institution:** Trinity College Dublin, Department of Mathematics  
**Degree:** MSc in High Performance Computing  
**Academic Year:** 2024-2025

---

## 🚀 Updated Overview

This project now consists of **two complementary components**:

1. **Original Intelligent KD-Tree Routing System** (this repo) – real-time data center selection using KD-Tree and live metrics.
2. **New MPI Parallel KD-Tree Module** – an HPC-focused resubmission that parallelizes KD-Tree queries using **MPI** and analyzes speedup, efficiency, and communication overhead.

Together, they demonstrate both **systems engineering** (real hardware, edge + distributed design) and **high-performance computing** (parallel algorithm design, MPI communication, and scalability analysis).

**Key Innovations:**

- Real-time performance metrics (latency, storage, throughput, distance) integrated with KD-Tree spatial indexing for intelligent routing.
- New **MPI-based parallel KD-Tree implementation** that distributes data and queries across multiple processes and profiles communication vs computation.

**Research Contributions:**

- Sub-millisecond query performance and up to **135× speedup** over brute-force methods in the original system.
- Detailed **MPI performance study** with speedup, efficiency, and communication overhead for different data sizes and process counts.

---

## 📋 Table of Contents

- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [New MPI Parallel KD-Tree Module](#-new-mpi-parallel-kd-tree-module)
- [Quick Start](#-quick-start)
- [System Implementations](#-system-implementations)
- [Performance Results](#-performance-results)
- [Technical Details](#-technical-details)
- [Research Context](#-research-context)
- [Citation](#-citation)
- [Contact](#-contact)

---

## ✨ Features

### Core Capabilities (Original System)

- **📊 Real-Time Metric Collection:** ICMP ping latency, FTP storage monitoring, throughput estimation.
- **🌲 KD-Tree Decision Engine:** Fast exact nearest-neighbor search in normalized 4D metric space.
- **⚡ Edge-Ready Architecture:** Lightweight deployment on OpenWRT routers with USB storage.
- **🔄 Distributed Scalability:** MapReduce-compatible design for enterprise-scale implementations.
- **📈 Advanced Visualization:** 3D scatter plots, radar charts, performance benchmarking.

### New HPC Capabilities (MPI Module)

- **🧮 Parallel KD-Tree with MPI:** Data and queries distributed over multiple MPI processes.
- **📉 Speedup & Efficiency Analysis:** Benchmarks for P = 2, 4, 8 and N = 500, 1000, 5000.
- **📡 Communication Profiling:** Explicit measurement of communication vs computation time.
- **📊 Automated Plotting:** Speedup and efficiency plots generated from a unified JSON benchmark file.

---

## 📁 Repository Structure

```
├── Local-System/                      # Single-node edge implementation
│   ├── main.py                        # Main orchestration and CLI
│   ├── collect_metrics.py             # ICMP ping & FTP storage collection
│   ├── kdtree_selector.py             # KD-Tree construction & querying
│   ├── normalize.py                   # Min-max normalization utilities
│   ├── benchmark.py                   # Performance benchmarking
│   ├── utils.py                       # Haversine distance calculation
│   ├── visualize.py                   # 3D visualization & radar charts
│   └── requirements.txt               # Python dependencies
├── Hadoop System/                     # Distributed MapReduce implementation
│   ├── distributed_kdtree_system.py   # Main distributed system
│   ├── data_generator.py              # Synthetic dataset generation
│   ├── compare_systems.py             # Local vs distributed benchmarks
│   ├── requirements.txt               # Distributed dependencies
│   └── data/                          # Generated test datasets
├── DIstributed-VM/                    # VM cluster validation
│   ├── kdtree_engine.py               # Simplified KD-Tree engine
│   ├── datacenter_topo.py             # Network topology simulation
│   ├── mininet_metrics.py             # Mininet integration
│   └── *.csv                          # Test datasets and results
├── parallel_mpi_kdtree/               # NEW: MPI-based KD-Tree (HPC resubmission)
│   ├── src/
│   │   ├── serial_kdtree.py           # Serial KD-Tree baseline
│   │   ├── parallel_kdtree_mpi.py     # MPI KD-Tree implementation
│   │   ├── performance_analysis.py    # Benchmarks & JSON export
│   │   └── visualization.py           # Speedup & efficiency plots
│   ├── tests/
│   │   └── test_data_comparison.py    # Correctness tests (serial vs parallel)
│   ├── results/
│   │   ├── benchmark_results.json     # Combined benchmark results (P,N)
│   │   ├── speedup_plot.png           # Speedup vs process count
│   │   └── efficiency_plot.png        # Efficiency vs process count
│   └── requirements.txt               # MPI-related dependencies
└── README.md                          # This documentation
```

> The `parallel_mpi_kdtree/` directory mirrors the standalone MPI repo  
> [`mpi-kdtree-data-center-selection`](https://github.com/seeshuraj/mpi-kdtree-data-center-selection)  
> and is integrated here as part of the final MSc thesis artefact.

---

## 🧮 New MPI Parallel KD-Tree Module

The **`parallel_mpi_kdtree/`** folder contains a clean, self-contained **MPI implementation of KD-Tree construction and querying** designed to satisfy the High Performance Computing resubmission requirements.

### Design Highlights

- **Data Distribution:**
  - Rank 0 holds the full dataset and uses `MPI_Scatter` to distribute `N` points across `P` processes.
  - Each process receives approximately `N / P` points.

- **Local KD-Tree Build:**
  - Each rank builds a local KD-Tree on its partition with no further communication.
  - This parallelizes the most expensive part of the algorithm.

- **Parallel Query Execution:**
  - Queries are broadcast using `MPI_Bcast` so all ranks can search their local trees.
  - Local results are gathered on rank 0 using `MPI_Gather` and merged to obtain the global nearest neighbours.

- **Performance Profiling:**
  - Communication and computation times are tracked separately.
  - A JSON benchmark file is generated for multiple `(N, P)` configurations and used to produce speedup and efficiency plots.

### How to Run the MPI Module

From the root of this repository:

```
cd parallel_mpi_kdtree

# (Optional) create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Run benchmarks for different process counts
mpiexec -n 2 python3 src/performance_analysis.py
mpiexec -n 4 python3 src/performance_analysis.py
mpiexec -n 8 python3 src/performance_analysis.py

# 2. Generate plots from the merged JSON results
python3 src/visualization.py

# 3. Run correctness tests (serial vs parallel KD-Tree)
mpiexec -n 4 python3 tests/test_data_comparison.py
```

This produces:

- `results/benchmark_results.json` – combined results for all runs  
- `results/speedup_plot.png` – speedup vs process count  
- `results/efficiency_plot.png` – efficiency vs process count  

---

## 🚀 Quick Start (Original System)

### Prerequisites

- **Python 3.10+**
- **Hardware:** OpenWRT routers with USB storage (for edge deployment)
- **Network:** FTP access to router storage, ICMP ping connectivity

### 1. Installation

```
# Clone repository
git clone https://github.com/seeshuraj/Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization.git
cd Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization

# Setup virtual environment
python3 -m venv kdtree_env
source kdtree_env/bin/activate  # Linux/Mac
# kdtree_env\Scripts\activate   # Windows

# Install dependencies for local edge system
pip install -r Local-System/requirements.txt
```

### 2. Quick Test (Local System)

```
cd Local-System
python main.py
```

This will:

- Collect metrics from configured routers  
- Build a normalized KD-Tree  
- Print the best-matching data center/router and its metrics  

### 3. Test Distributed System

```
cd "Hadoop System"

# Generate synthetic datasets
python data_generator.py

# Run distributed KD-Tree system
python distributed_kdtree_system.py

# Compare local vs distributed performance
python compare_systems.py
```

---

## 🖥️ System Implementations (Summary)

### Local Edge System

Target: OpenWRT routers, Raspberry Pi, resource-constrained devices.

```
cd Local-System

# Real-time metric collection
python collect_metrics.py

# Full system with visualization
python main.py

# Performance benchmarking
python benchmark.py

# Generate visualizations
python visualize.py
```

### Distributed Cloud System (Hadoop)

Target: Hadoop clusters, enterprise deployments, 10K+ nodes.

```
cd "Hadoop System"
python distributed_kdtree_system.py
```

---

## 📊 Performance Results (High Level)

### Original KD-Tree System

| System Type      | Nodes  | Build Time | Query Time | Memory Usage | Success Rate |
|------------------|--------|-----------:|-----------:|-------------:|-------------:|
| Local KD-Tree    | 100    | 8.3ms      | 0.5ms      | ~2MB         | 100%         |
| Local KD-Tree    | 1,000  | 24.7ms     | 1.1ms      | ~15MB        | 100%         |
| Local KD-Tree    | 10,000 | 129.6ms    | 6.3ms      | ~120MB       | 100%         |
| Distributed      | 10,000 | 1.7s       | ~50ms      | Distributed  | 100%         |
| Brute Force      | 10,000 | <1ms build | 850ms      | ~8MB         | 100%         |

**Key Achievement:** Up to **135× speedup** over brute-force selection while maintaining exact nearest-neighbour accuracy.

### MPI KD-Tree (HPC Module)

For each data size `N ∈ {500, 1000, 5000}` and process count `P ∈ {2, 4, 8}`, the MPI module reports:

- **Speedup** \( S = T_{serial} / T_{parallel} \)  
- **Parallel efficiency** \( E = S / P × 100\% \)  
- **Communication overhead** as a percentage of total runtime  

These are visualized in:

- `parallel_mpi_kdtree/results/speedup_plot.png`  
- `parallel_mpi_kdtree/results/efficiency_plot.png`  

The results show classic HPC behaviour:

- Small `N` → communication dominates, speedup < 1  
- Larger `N` → computation dominates more, speedup improves, but efficiency drops as `P` grows due to increasing communication overhead  

---

## 🔧 Technical Details (Selected)

### Metric Collection (Original System)

```
def get_latency(ip):
    cmd = ["ping", "-c", "5", ip]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse average RTT from output

def get_storage_via_ftp(ip):
    ftp = FTP(ip)
    ftp.login('anonymous', '')
    ftp.cwd('shares/USB_Storage')
    # Calculate used storage by file crawling
```

### KD-Tree Implementation (Original)

```
from scipy.spatial import KDTree

# Normalize metrics to[1]
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
best_router = routers[index]
```

### MPI KD-Tree (Parallel Outline)

In `parallel_mpi_kdtree/src/parallel_kdtree_mpi.py`:

1. Initialize MPI communicator and ranks  
2. Scatter dataset from rank 0 to all ranks  
3. Build a local KD-Tree on each rank  
4. Broadcast queries and gather local results  
5. Merge candidates on rank 0 to produce global k-nearest neighbours  

---

## 🎓 Research Context

This repository supports the MSc thesis:

> **"Intelligent Data Center Selection using KD-Tree Based Performance and Distance Optimization"**  
> Trinity College Dublin, Department of Mathematics  
> MSc in High Performance Computing

The integration of the MPI module demonstrates:

1. Parallel algorithm design and MPI implementation  
2. Quantitative analysis of speedup, efficiency, and communication overhead  
3. Understanding when parallelization helps or hurts, based on problem size and communication cost  

---

## 📖 Citation

```
@mastersthesis{bhoopalan2025kdtree,
    title        = {Intelligent Data Center Selection using KD-Tree Based Performance and Distance Optimization},
    author       = {Seeshuraj Bhoopalan},
    school       = {Trinity College Dublin},
    department   = {Department of Mathematics},
    year         = {2025},
    degree       = {MSc in High Performance Computing},
    supervisor   = {Kirk M. Soodhalter},
    url          = {https://github.com/seeshuraj/Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization}
}
```

---

## 📞 Contact

**Seeshuraj Bhoopalan**  
📧 **Email:** bhoopals@tcd.ie  
🏛️ **Institution:** Trinity College Dublin  
🔗 **LinkedIn:** https://www.linkedin.com/in/seeshuraj-b-051260122/

---

## 🏆 Acknowledgments

- **Supervisor:** Kirk M. Soodhalter  
- **Institution:** Trinity College Dublin, Department of Mathematics  
- **HPC & Open-Source Community:** For tools, libraries, and foundational algorithms  
```
