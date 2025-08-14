# visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_metrics_bar(df):
    df_plot = df[["router_ip", "latency_ms", "throughput_mbps", "free_storage_gb"]]
    df_plot.set_index("router_ip").plot(kind="bar", figsize=(10, 5))
    plt.title("Router Performance Metrics")
    plt.ylabel("Value")
    plt.xlabel("Router IP")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_normalized_radar(df):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = ['norm_storage', 'norm_distance', 'norm_score']
    N = len(categories)

    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, row in df.iterrows():
        values = row[categories].tolist()
        values += values[:1]  # repeat the first value to close the circle
        ax.plot(angles, values, label=row["router_ip"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Normalized KD-Tree Input Features")
    ax.legend(loc='upper right')
    plt.show()

def plot_3d_kdtree(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        df["norm_storage"],
        df["norm_distance"],
        df["norm_score"],
        c='b', marker='o'
    )
    ax.set_xlabel('Norm Storage')
    ax.set_ylabel('Norm Distance')
    ax.set_zlabel('Norm Score')
    plt.title("KD-Tree Feature Space (3D)")
    plt.show()
