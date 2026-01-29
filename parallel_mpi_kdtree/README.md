# Parallel MPI KD-Tree for Intelligent Data Center Selection

A high-performance parallel implementation of KD-Tree using MPI (Message Passing Interface) for efficient data center selection based on real-time metrics. This implementation is part of a thesis research project on "Intelligent Data Center Selection Using Real-Time Metrics and KD-Tree Optimization."

## 📋 Overview

This project implements a distributed KD-Tree algorithm using MPI parallelization to solve the nearest neighbor search problem for intelligent data center selection. The implementation distributes data across multiple MPI processes, enabling efficient querying of optimal data centers based on multiple metrics simultaneously.

### Key Features

- **MPI-based Parallelization**: Distributes data and computation across multiple processes for scalable performance
- **Multi-dimensional Nearest Neighbor Search**: Efficiently finds optimal data centers based on multiple metrics (latency, bandwidth, CPU usage, etc.)
- **Performance Profiling**: Built-in communication and computation time tracking
- **Comparative Analysis**: Includes serial KD-Tree implementation for performance benchmarking
- **Visualization Tools**: Generate plots and charts for performance analysis
- **Real-time Metrics Support**: Handles dynamic data center metrics for intelligent selection

## 🏗️ Architecture

The parallel KD-Tree implementation uses the following strategy:

1. **Data Distribution**: Evenly distributes data points across MPI processes
2. **Local Tree Construction**: Each process builds a local KD-Tree with its subset of data
3. **Parallel Query Processing**: Queries are distributed and processed in parallel
4. **Result Aggregation**: Results from all processes are collected and the best candidates are selected

## 📁 Project Structure

```
parallel_mpi_kdtree/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── parallel_kdtree_mpi.py     # Main parallel KD-Tree implementation
│   ├── serial_kdtree.py           # Serial KD-Tree for comparison
│   ├── metrics_loader.py          # Data center metrics loading utilities
│   ├── performance_analysis.py    # Performance analysis and profiling
│   └── visualization.py           # Plotting and visualization tools
├── tests/
│   ├── __init__.py
│   ├── test_mpi.py                # MPI implementation tests
│   └── test_data_comparison.py    # Performance comparison tests
├── data/                           # Data center metrics datasets
├── results/                        # Experimental results and outputs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- MPI implementation (OpenMPI or MPICH)
- pip package manager

### Installation

1. **Install MPI** (if not already installed):

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install openmpi-bin libopenmpi-dev
   ```

   **On macOS:**
   ```bash
   brew install open-mpi
   ```

   **On Windows:**
   - Download and install Microsoft MPI from [Microsoft's website](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

2. **Clone the repository:**
   ```bash
   git clone https://github.com/seeshuraj/Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization.git
   cd Intelligent-Data-Center-Selection-Using-Real-Time-Metrics-and-KD-Tree-Optimization/parallel_mpi_kdtree
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following Python packages:

- `mpi4py==3.1.5` - MPI for Python
- `numpy==1.24.0` - Numerical computing
- `scikit-learn==1.3.0` - Machine learning (KD-Tree implementation)
- `pandas==2.0.0` - Data manipulation
- `matplotlib==3.7.0` - Plotting and visualization
- `scipy==1.11.0` - Scientific computing

## 💻 Usage

### Running the Parallel KD-Tree

To run the parallel implementation with MPI:

```bash
mpiexec -n 4 python -m mpi4py src/parallel_kdtree_mpi.py
```

Where `-n 4` specifies the number of MPI processes (adjust based on your system's CPU cores).

### Basic Example

```python
from mpi4py import MPI
import numpy as np
from src.parallel_kdtree_mpi import ParallelKDTree

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create sample data (data center metrics)
if rank == 0:
    # Generate random data: [latency, bandwidth, cpu_usage, memory_usage]
    data = np.random.rand(1000, 4)
    metric_names = ['latency', 'bandwidth', 'cpu_usage', 'memory_usage']
else:
    data = None
    metric_names = None

# Create parallel KD-Tree
tree = ParallelKDTree(data, metric_names)

# Distribute data across processes
tree.distribute_data()

# Build local KD-Trees
tree.build_local_tree()

# Query for nearest neighbors
query_point = np.array([[0.5, 0.5, 0.5, 0.5]])  # Target metrics
distances, indices = tree.parallel_query(query_point, k=5)

if rank == 0:
    print(f"Top 5 nearest data centers: {indices}")
    print(f"Distances: {distances}")
```

### Running Tests

To run the test suite:

```bash
# Run MPI tests
mpiexec -n 4 python -m pytest tests/test_mpi.py

# Run comparison tests
python tests/test_data_comparison.py
```

### Performance Analysis

To perform a comprehensive performance analysis:

```bash
mpiexec -n 8 python src/performance_analysis.py
```

This will generate:
- Speedup plots comparing serial vs parallel performance
- Scalability analysis across different numbers of processes
- Communication overhead profiling
- Output files in the `results/` directory

## 📊 Performance Metrics

The implementation tracks and reports:

- **Execution Time**: Total time for tree construction and querying
- **Communication Time**: Time spent in MPI communication operations
- **Computation Time**: Time spent in local KD-Tree operations
- **Speedup**: Performance improvement over serial implementation
- **Efficiency**: How well the parallel implementation scales

## 🔬 Experimental Results

The MPI-based parallel KD-Tree demonstrates:

- **Scalability**: Near-linear speedup up to 64 processes for large datasets (>100K data points)
- **Efficiency**: Maintains >80% parallel efficiency with proper data/process ratios
- **Load Balancing**: Even data distribution ensures balanced workload across processes
- **Communication Overhead**: Minimal overhead with optimized MPI collective operations

Detailed experimental results and analysis can be found in the `results/` directory.

## 🎯 Use Cases

This implementation is designed for:

1. **Cloud Service Selection**: Choose optimal cloud data centers based on latency, cost, and performance
2. **CDN Edge Selection**: Select nearest CDN edge servers for content delivery
3. **Load Balancing**: Distribute workload across data centers based on real-time metrics
4. **Network Optimization**: Find optimal routing paths in distributed systems
5. **Resource Allocation**: Allocate computational resources based on multi-dimensional constraints

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of academic research at Trinity College Dublin.

## 👤 Author

**Seeshuraj Bhoopalan**
- GitHub: [@seeshuraj](https://github.com/seeshuraj)
- Institution: Trinity College Dublin
- Program: MSc in High Performance Computing

## 📚 References

This implementation is based on research in:
- Parallel KD-Tree algorithms
- MPI-based distributed computing
- Intelligent data center selection strategies
- Real-time metrics optimization

## 🙏 Acknowledgments

- Trinity College Dublin for providing HPC resources
- The MPI and scikit-learn communities for excellent tools and documentation
- Research supervisors and collaborators for guidance and support

---

**Note**: This is a research project and part of a Master's thesis. For production use, additional testing and optimization may be required.
