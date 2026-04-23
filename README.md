# Health Insurance Cross-Sell Data Mining Pipeline

Transforming raw insurance data into actionable business intelligence using unsupervised learning, fuzzy logic, and genetic search.

## 📌 Project Overview
This repository contains a full-stack Data Mining pipeline crafted strictly according to the academic rubrics. It investigates a subset of Kaggle's Health Insurance dataset to predict and profile customers who are likely to cross-buy vehicle insurance.

The project encompasses:
1. **Exploratory Data Analysis** (5 robust visualizations).
2. **Data Preprocessing** (Outlier clipping, Missing value interpolation, Standard Scaling, and Dimensionality Reduction via PCA).
3. **Clustering Analysis** (K-Medoids & Hierarchical clustering evaluated via Silhouette and Elbow scores).
4. **Fuzzy Logic Expert System** (A 9-rule Mamdani system determining target priority using Clustering labels).
5. **Genetic Algorithm Optimization** (Evolutionary feature subset selection).
6. **Streamlit UI** (An interactive frontend for immediate real-world deployment).

## 📁 Repository Structure
```text
ProjectDemo/
│
├── data/
│   ├── raw_data.csv                 # Original raw dataset
│   └── cleaned_data.csv             # The normalized subset used across the tools
│
├── notebooks/
│   └── Final_Project_Submission.ipynb # 🚀 EXECUTABLE MAIN NOTEBOOK (All 8 Sections)
│
├── artifacts/
│   └── ...                          # Pre-trained Encoders, Scalers, and Models
│
├── ml_model/
│   ├── kmedoids.py                  # Custom K-Medoid class implementation
│   ├── predict.py                   # Production inference logic
│   └── train_model.py               # Headless training pipeline
│
├── ui/
│   └── app.py                       # Advanced UI Dashboard via Streamlit
│
└── utils/
    └── data_processing.py           # Robust, reusable data cleansing logic
```

## 🚀 Quick Setup & Execution

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. View The Academic Report
The `notebooks/Final_Project_Submission.ipynb` contains the highly-detailed required structure (Sections 1 through 8) completely written in Markdown.
Simply launch Visual Studio Code or Jupyter, select the Python 3 Kernel, and hit **Run All**.

### 3. Launch The Web App (Bonus)
A highly professional Graphical Interface allows rapid testing of the models:
```powershell
streamlit run ui/app.py
```

## 🧠 System Pipeline
1. **Raw Inputs** -> User feeds demographic strings.
2. **Cleaning** -> Categoricals are converted to Ints; continuous values hit `StandardScaler`.
3. **Clustering Inference** -> Features hit the `KMedoids` algorithm to return a Segment Class (e.g., *Hot Lead*).
4. **Fuzzy Engine** -> Segment Class + Age dictate Business Priority Strategy.
5. **UI Display** -> Streamlit displays the output in real-time.

---
*Created as part of the Data Mining University requirements.*
