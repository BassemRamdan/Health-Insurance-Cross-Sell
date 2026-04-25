import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    """
    Cleans raw dataframe by dropping 'id', filling missing values, and handling outliers.
    """
    df_clean = df.copy()
    
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
        
    for col in ['Age', 'Annual_Premium', 'Vintage']:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    if 'Annual_Premium' in df_clean.columns:
        Q1 = df_clean['Annual_Premium'].quantile(0.25)
        Q3 = df_clean['Annual_Premium'].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df_clean['Annual_Premium'] = df_clean['Annual_Premium'].clip(lower=lower_limit, upper=upper_limit)
        
    if 'Vehicle_Age' in df_clean.columns:
        vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
        df_clean['Vehicle_Age'] = df_clean['Vehicle_Age'].replace(vehicle_age_map).infer_objects(copy=False)
            
    return df_clean

def encode_and_scale(df, is_training=True, artifacts_dir='artifacts'):
    """
    Encodes categorical features and scales numerical features.
    """
    df_processed = df.copy()
    
    if is_training:
        le_gender = LabelEncoder()
        df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
        joblib.dump(le_gender, os.path.join(artifacts_dir, 'le_gender.joblib'))
        
        le_damage = LabelEncoder()
        df_processed['Vehicle_Damage'] = le_damage.fit_transform(df_processed['Vehicle_Damage'])
        joblib.dump(le_damage, os.path.join(artifacts_dir, 'le_damage.joblib'))
    else:
        le_gender = joblib.load(os.path.join(artifacts_dir, 'le_gender.joblib'))
        df_processed['Gender'] = le_gender.transform(df_processed['Gender'])
        
        le_damage = joblib.load(os.path.join(artifacts_dir, 'le_damage.joblib'))
        df_processed['Vehicle_Damage'] = le_damage.transform(df_processed['Vehicle_Damage'])

    # Get the correct column order expected by the scaler
    raw_cols_path = os.path.join(artifacts_dir, 'raw_columns.joblib')
    if os.path.exists(raw_cols_path):
        cols_to_scale = joblib.load(raw_cols_path)
        # Ensure we only scale columns that actually exist in the df (in case Response is in raw_cols but not df)
        cols_to_scale = [c for c in cols_to_scale if c in df_processed.columns]
    else:
        cols_to_scale = [c for c in df_processed.columns if c != 'Response']
    
    if len(cols_to_scale) > 0:
        if is_training:
            scaler = StandardScaler()
            df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
            joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
            joblib.dump(cols_to_scale, os.path.join(artifacts_dir, 'raw_columns.joblib'))
        else:
            scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
            # Reorder df to match scaler training exactly
            df_processed = df_processed[cols_to_scale]
            scaled_vals = scaler.transform(df_processed)
            df_processed = pd.DataFrame(scaled_vals, columns=cols_to_scale, index=df_processed.index)
        
    return df_processed
