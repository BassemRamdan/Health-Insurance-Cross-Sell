import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics.pairwise import pairwise_distances
import skfuzzy as fuzz
from skfuzzy import control as ctrl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_processing import encode_and_scale
# Import the class so joblib can unpickle it
from ml_model.kmedoids import SimpleKMedoids

def get_fuzzy_score(age_val, cluster_val):
    age = ctrl.Antecedent(np.arange(18, 86, 1), 'age')
    cluster_label = ctrl.Antecedent(np.arange(0, 3, 1), 'cluster')
    action_score = ctrl.Consequent(np.arange(0, 11, 1), 'action_score')

    age['young'] = fuzz.trimf(age.universe, [18, 18, 35])
    age['middle'] = fuzz.trimf(age.universe, [30, 45, 60])
    age['senior'] = fuzz.trimf(age.universe, [55, 86, 86])

    cluster_label['safe'] = fuzz.trimf(cluster_label.universe, [0, 0, 0.5])
    cluster_label['hot'] = fuzz.trimf(cluster_label.universe, [0.5, 1, 1.5])
    cluster_label['core'] = fuzz.trimf(cluster_label.universe, [1.5, 2, 2])

    action_score.automf(names=['ignore', 'monitor', 'target'])

    rules = [
        ctrl.Rule(age['young'] & cluster_label['safe'], action_score['ignore']),
        ctrl.Rule(age['young'] & cluster_label['hot'], action_score['target']),
        ctrl.Rule(age['young'] & cluster_label['core'], action_score['monitor']),
        ctrl.Rule(age['middle'] & cluster_label['safe'], action_score['monitor']),
        ctrl.Rule(age['middle'] & cluster_label['hot'], action_score['target']),
        ctrl.Rule(age['middle'] & cluster_label['core'], action_score['target']),
        ctrl.Rule(age['senior'] & cluster_label['safe'], action_score['ignore']),
        ctrl.Rule(age['senior'] & cluster_label['hot'], action_score['monitor']),
        ctrl.Rule(age['senior'] & cluster_label['core'], action_score['monitor']),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    
    sim.input['age'] = float(age_val)
    sim.input['cluster'] = float(cluster_val)
    sim.compute()
    
    score = sim.output['action_score']
    if score >= 6.5: action = "TARGET"
    elif score >= 4: action = "MONITOR"
    else: action = "IGNORE"
    
    return float(score), action

def predict_pipeline(customer_dict):
    """
    Predicts K-Medoid and Hierarchical cluster segments, and evaluates Fuzzy Logic.
    """
    df = pd.DataFrame([customer_dict])
    
    if 'Vehicle_Age' in df.columns:
        vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
        df['Vehicle_Age'] = df['Vehicle_Age'].replace(vehicle_age_map).infer_objects(copy=False)
            
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    df_processed = encode_and_scale(df, is_training=False, artifacts_dir=artifacts_dir)
    
    if 'Response' in df_processed.columns:
        df_processed = df_processed.drop('Response', axis=1)
        
    kmedoids_path = os.path.join(artifacts_dir, 'kmedoids_model.joblib')
    hier_path = os.path.join(artifacts_dir, 'hierarchical_centroids.joblib')
    
    if not os.path.exists(kmedoids_path) or not os.path.exists(hier_path):
        return {"error": "Models not found. Please run train_model.py first."}
        
    kmedoids_model = joblib.load(kmedoids_path)
    kmedoid_cluster = kmedoids_model.predict(df_processed.values)[0]
    
    hier_centroids = joblib.load(hier_path)
    distances = pairwise_distances(df_processed.values, hier_centroids)
    hier_cluster = np.argmin(distances, axis=1)[0]
    
    f_score, f_action = get_fuzzy_score(customer_dict.get('Age', 25), kmedoid_cluster)
    
    return {
        "kmedoid_cluster": int(kmedoid_cluster),
        "hierarchical_cluster": int(hier_cluster),
        "fuzzy_score": f_score,
        "fuzzy_action": f_action
    }

if __name__ == '__main__':
    # Test example
    sample_customer = {
        'Gender': 'Male',
        'Age': 25,
        'Driving_License': 1,
        'Region_Code': 28.0,
        'Previously_Insured': 0,
        'Vehicle_Age': '< 1 Year',
        'Vehicle_Damage': 'Yes',
        'Annual_Premium': 38000,
        'Policy_Sales_Channel': 152.0,
        'Vintage': 100
    }
    print("Prediction Result:", predict_pipeline(sample_customer))
