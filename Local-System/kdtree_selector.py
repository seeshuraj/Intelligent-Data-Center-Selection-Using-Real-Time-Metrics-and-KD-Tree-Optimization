# kdtree_selector.py

import numpy as np
from scipy.spatial import KDTree

def build_kdtree(dataframe, feature_columns):
    """
    Build a KDTree from the given dataframe and specified feature columns.
    
    Args:
        dataframe (pd.DataFrame): The dataframe containing the feature columns.
        feature_columns (list): List of column names to use as KD-Tree dimensions.

    Returns:
        tree (KDTree), feature matrix (np.ndarray)
    """
    features = dataframe[feature_columns].values
    tree = KDTree(features)
    return tree, features

def query_kdtree(tree, dataframe, query_vector, k=1):
    """
    Query the KDTree with a given vector and return the closest router(s).
    
    Args:
        tree (KDTree): The built KD-Tree.
        dataframe (pd.DataFrame): The original dataframe to retrieve the matched row.
        query_vector (list or np.ndarray): The user preference vector (normalized).
        k (int): Number of nearest neighbors to return.

    Returns:
        result_df (pd.DataFrame): The best matching rows from the dataframe.
    """
    dist, idx = tree.query(np.array(query_vector).reshape(1, -1), k=k)
    return dataframe.iloc[idx]
