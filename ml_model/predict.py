import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def run_pipeline(customer_dict):
    """
    Takes a new customer as a dictionary.
    Returns cluster number, interest score, and recommendation based on Bassem's logic.
    """
    # 1. Load artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    le_gender = joblib.load(os.path.join(artifacts_dir, 'le_gender.joblib'))
    le_damage = joblib.load(os.path.join(artifacts_dir, 'le_damage.joblib'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
    kmedoids_model = joblib.load(os.path.join(artifacts_dir, 'kmedoids_model.joblib'))
    hier_centroids = joblib.load(os.path.join(artifacts_dir, 'hierarchical_centroids.joblib'))
    best_features = joblib.load(os.path.join(artifacts_dir, 'ga_features.joblib'))
    raw_columns = joblib.load(os.path.join(artifacts_dir, 'raw_columns.joblib'))

    # Save original values for fuzzy system (real values not scaled)
    original_age = float(customer_dict.get('Age', 30))
    original_premium = float(customer_dict.get('Annual_Premium', 35000))
    original_prev_ins = float(customer_dict.get('Previously_Insured', 0))
    original_damage = 1.0 if customer_dict.get('Vehicle_Damage', 'No') == 'Yes' else 0.0

    # Step 1: Make a one row dataframe
    record = pd.DataFrame([customer_dict])

    # Handle missing/default values if any
    for col in raw_columns:
        if col not in record.columns:
            if col == 'Age': record[col] = 30
            elif col == 'Annual_Premium': record[col] = 35000
            elif col == 'Vintage': record[col] = 150
            elif col == 'Gender': record[col] = 'Male'
            elif col == 'Vehicle_Age': record[col] = '1-2 Year'
            elif col == 'Vehicle_Damage': record[col] = 'No'
            else: record[col] = 0.0

    # Step 2: Encode text columns using ALREADY FITTED encoders
    record['Gender'] = le_gender.transform(record['Gender'])
    record['Vehicle_Damage'] = le_damage.transform(record['Vehicle_Damage'])
    record['Vehicle_Age'] = record['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}).fillna(0)

    # Make sure columns are in correct order for scaler
    record = record[raw_columns]

    # Step 3: Scale numeric columns using ALREADY FITTED scaler
    scaled_values = scaler.transform(record)
    record_scaled = pd.DataFrame(scaled_values, columns=raw_columns)

    # Step 4: Keep only GA-selected features for cluster prediction
    record_for_cluster = record_scaled[best_features].values

    # Step 5: Predict clusters using already-trained K-Medoid
    cluster_label = int(kmedoids_model.predict(record_for_cluster)[0])
    
    # Predict Hierarchical (nearest centroid)
    from sklearn.metrics.pairwise import euclidean_distances
    hier_dist = euclidean_distances(record_for_cluster, hier_centroids)
    hier_cluster = int(np.argmin(hier_dist, axis=1)[0])

    # Step 6: Define Fuzzy System based on 13 rules from Bassem's notebook
    age_var = ctrl.Antecedent(np.arange(20, 86, 1), 'age')
    premium_var = ctrl.Antecedent(np.arange(2000, 500000, 1000), 'premium')
    prev_var = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'previously_insured')
    damage_var = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'vehicle_damage')
    score_var = ctrl.Consequent(np.arange(0, 101, 1), 'interest_score')

    age_var['young'] = fuzz.trimf(age_var.universe, [20, 20, 35])
    age_var['middle'] = fuzz.trimf(age_var.universe, [30, 45, 60])
    age_var['senior'] = fuzz.trimf(age_var.universe, [50, 85, 85])

    premium_var['low'] = fuzz.trimf(premium_var.universe, [2000, 2000, 30000])
    premium_var['medium'] = fuzz.trimf(premium_var.universe, [25000, 40000, 55000])
    premium_var['high'] = fuzz.trimf(premium_var.universe, [50000, 500000, 500000])

    prev_var['not_insured'] = fuzz.trimf(prev_var.universe, [0, 0, 0.2])
    prev_var['insured'] = fuzz.trimf(prev_var.universe, [0.8, 1, 1])

    damage_var['no_damage'] = fuzz.trimf(damage_var.universe, [0, 0, 0.2])
    damage_var['had_damage'] = fuzz.trimf(damage_var.universe, [0.8, 1, 1])

    score_var['very_low'] = fuzz.trimf(score_var.universe, [0, 0, 25])
    score_var['low'] = fuzz.trimf(score_var.universe, [15, 30, 45])
    score_var['medium'] = fuzz.trimf(score_var.universe, [40, 55, 70])
    score_var['high'] = fuzz.trimf(score_var.universe, [60, 75, 90])
    score_var['very_high'] = fuzz.trimf(score_var.universe, [85, 100, 100])

    # 13 Rules from Bassem's notebook
    rules = [
        ctrl.Rule(damage_var['had_damage'] & prev_var['not_insured'], score_var['very_high']),
        ctrl.Rule(prev_var['insured'], score_var['very_low']),
        ctrl.Rule(damage_var['no_damage'] & prev_var['not_insured'], score_var['medium']),
        ctrl.Rule(damage_var['had_damage'] & age_var['middle'], score_var['very_high']),
        ctrl.Rule(damage_var['had_damage'] & age_var['young'], score_var['high']),
        ctrl.Rule(damage_var['had_damage'] & age_var['senior'], score_var['medium']),
        ctrl.Rule(damage_var['no_damage'] & age_var['senior'], score_var['low']),
        ctrl.Rule(damage_var['had_damage'] & premium_var['high'], score_var['very_high']),
        ctrl.Rule(damage_var['had_damage'] & premium_var['medium'], score_var['high']),
        ctrl.Rule(damage_var['had_damage'] & premium_var['low'], score_var['medium']),
        ctrl.Rule(damage_var['no_damage'] & premium_var['low'], score_var['low']),
        ctrl.Rule(damage_var['no_damage'] & premium_var['high'], score_var['medium']),
        ctrl.Rule(age_var['middle'] & prev_var['not_insured'] & premium_var['high'], score_var['very_high'])
    ]

    interest_ctrl = ctrl.ControlSystem(rules)
    interest_sim = ctrl.ControlSystemSimulation(interest_ctrl)

    # Step 7: Calculate Fuzzy Score
    try:
        interest_sim.input['age'] = float(original_age)
        interest_sim.input['premium'] = float(original_premium)
        interest_sim.input['previously_insured'] = float(original_prev_ins)
        interest_sim.input['vehicle_damage'] = float(original_damage)
        interest_sim.compute()
        score = round(interest_sim.output['interest_score'], 1)
    except Exception as e:
        print("Fuzzy Error:", e)
        score = 50.0

    if score >= 65:
        recommendation = 'HIGH PRIORITY - Call immediately'
    elif score >= 40:
        recommendation = 'MEDIUM PRIORITY - Send campaign'
    else:
        recommendation = 'LOW PRIORITY - Skip'

    return {
        'kmedoid_cluster': cluster_label,
        'hierarchical_cluster': hier_cluster,
        'fuzzy_score': score,
        'fuzzy_action': recommendation
    }

# Keep this for compatibility with app.py if needed, or point app.py to run_pipeline
predict_pipeline = run_pipeline

if __name__ == '__main__':
    c3 = {
        'Gender': 'Male', 'Age': 44, 'Driving_License': 1, 'Region_Code': 15.0,
        'Previously_Insured': 0, 'Vehicle_Age': '1-2 Year', 'Vehicle_Damage': 'Yes',
        'Annual_Premium': 58000, 'Policy_Sales_Channel': 26.0, 'Vintage': 95
    }
    print(run_pipeline(c3))
