import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ml_model.kmedoids import SimpleKMedoids
from sklearn.cluster import AgglomerativeClustering

def train_and_save_models():
    print("Loading data...")
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_data.csv')
    df = pd.read_csv(data_path)
    
    print("Preprocessing...")
    df_clean = df.copy()
    if 'id' in df_clean.columns:
        df_clean.drop('id', axis=1, inplace=True)
        
    for col in ['Age', 'Annual_Premium', 'Vintage']:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
    Q1 = df_clean['Annual_Premium'].quantile(0.25)
    Q3 = df_clean['Annual_Premium'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean['Annual_Premium'] = df_clean['Annual_Premium'].clip(lower=lower, upper=upper)
    
    vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df_clean['Vehicle_Age'] = df_clean['Vehicle_Age'].map(vehicle_age_map).fillna(0)
    
    le_gender = LabelEncoder()
    df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])
    
    le_damage = LabelEncoder()
    df_clean['Vehicle_Damage'] = le_damage.fit_transform(df_clean['Vehicle_Damage'])
    
    if 'Response' in df_clean.columns:
        X_raw = df_clean.drop('Response', axis=1)
    else:
        X_raw = df_clean
        
    # Scale ALL features
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_raw)
    df_scaled_full = pd.DataFrame(X_scaled_full, columns=X_raw.columns)
    
    # GA selected features
    best_features = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage']
    X_ga = df_scaled_full[best_features].values
    
    print("Training K-Medoids on GA selected features...")
    # Take a sample for K-Medoids training to save time
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_ga), 1500, replace=False)
    X_train = X_ga[sample_idx]
    
    kmedoids_model = SimpleKMedoids(n_clusters=3, max_iter=100)
    kmedoids_model.fit(X_train)
    
    print("Training Hierarchical Clustering...")
    hier_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hier_model.fit(X_train)
    
    # Compute centroids for hierarchical so we can predict later
    centroids = []
    for i in range(3):
        cluster_points = X_train[hier_model.labels_ == i]
        if len(cluster_points) > 0:
            centroids.append(cluster_points.mean(axis=0))
        else:
            centroids.append(np.zeros(X_train.shape[1]))
    hier_centroids = np.array(centroids)
    
    print("Saving artifacts...")
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(le_gender, os.path.join(artifacts_dir, 'le_gender.joblib'))
    joblib.dump(le_damage, os.path.join(artifacts_dir, 'le_damage.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
    joblib.dump(kmedoids_model, os.path.join(artifacts_dir, 'kmedoids_model.joblib'))
    joblib.dump(hier_centroids, os.path.join(artifacts_dir, 'hierarchical_centroids.joblib'))
    
    # Save the selected feature names so predict.py knows what to extract
    joblib.dump(best_features, os.path.join(artifacts_dir, 'ga_features.joblib'))
    # Save the column order for scaling
    joblib.dump(list(X_raw.columns), os.path.join(artifacts_dir, 'raw_columns.joblib'))
    
    print("All artifacts saved successfully!")

if __name__ == '__main__':
    train_and_save_models()
