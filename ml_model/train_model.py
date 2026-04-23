import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_processing import clean_data, encode_and_scale
from ml_model.kmedoids import SimpleKMedoids

def train_and_save():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_data.csv')
    df = pd.read_csv(data_path)
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print("Encoding and scaling...")
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    df_processed = encode_and_scale(df_clean, is_training=True, artifacts_dir=artifacts_dir)
    
    if 'Response' in df_processed.columns:
        X = df_processed.drop('Response', axis=1)
    else:
        X = df_processed
        
    print("Sampling 5000 rows for memory efficiency...")
    sample_size = 5000
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    
    print("Training K-Medoids...")
    kmedoids_model = SimpleKMedoids(n_clusters=3, max_iter=50, random_state=42)
    kmedoids_model.fit(X_sample.values)
    
    kmedoids_path = os.path.join(artifacts_dir, 'kmedoids_model.joblib')
    joblib.dump(kmedoids_model, kmedoids_path)
    
    print("Training Hierarchical Clustering...")
    hier_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hier_labels = hier_model.fit_predict(X_sample.values)
    
    print("Extracting Hierarchical centroids...")
    hier_centroids = []
    for i in range(3):
        cluster_points = X_sample.values[hier_labels == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            hier_centroids.append(centroid)
        else:
            hier_centroids.append(np.zeros(X_sample.values.shape[1]))
            
    hier_centroids = np.array(hier_centroids)
    hier_path = os.path.join(artifacts_dir, 'hierarchical_centroids.joblib')
    joblib.dump(hier_centroids, hier_path)
    
    print(f"Models saved successfully to {artifacts_dir}")

if __name__ == '__main__':
    train_and_save()
