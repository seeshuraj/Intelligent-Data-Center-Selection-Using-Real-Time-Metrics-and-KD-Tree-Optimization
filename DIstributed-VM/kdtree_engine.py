#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import math

def normalize(series):
    """Min–max normalize a pandas Series to [0,1]."""
    minv, maxv = series.min(), series.max()
    if maxv == minv:
        return pd.Series(0.5, index=series.index)
    return (series - minv) / (maxv - minv)

def haversine(lat1, lon1, lat2, lon2):
    """Compute Haversine distance (km) between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def build_and_query(df, query_ip=None):
    # Normalize metrics
    df['norm_latency'] = normalize(df['latency_ms'])
    df['norm_free_storage'] = normalize(df['free_storage_gb'])
    df['norm_distance'] = normalize(df['distance_km'])

    features = df[['norm_latency','norm_free_storage','norm_distance']].values
    tree = KDTree(features)

    if query_ip:
        row = df[df['ip_address']==query_ip].iloc[0]
        q = np.array([[row.norm_latency, row.norm_free_storage, row.norm_distance]])
    else:
        # default: use median values
        q = np.median(features, axis=0).reshape(1,-1)

    dist, idx = tree.query(q, k=1)
    best = df.iloc[idx[0][0]]
    return best, dist[0][0]

def main():
    parser = argparse.ArgumentParser(
        description='KD-Tree engine for data center selection')
    parser.add_argument('--input', '-i', required=True,
                        help='CSV file of metrics')
    parser.add_argument('--query-ip', '-q', default=None,
                        help='IP address to use as query vector')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Compute geographic distances if missing
    if 'distance_km' not in df.columns:
        lat0, lon0 = df.loc[0,['latitude','longitude']]
        df['distance_km'] = df.apply(
            lambda r: haversine(lat0, lon0, r.latitude, r.longitude),
            axis=1
        )

    best, d = build_and_query(df, args.query_ip)
    print('Selected Data Center:')
    print(best[['node_id','ip_address','latency_ms',
                'throughput_mbps','free_storage_gb','distance_km']])
    print(f'Distance in feature space: {d:.4f}')

if __name__ == '__main__':
    main()
