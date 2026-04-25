import json

notebook_path = r'C:\ANU\DataMining\InsureDx\notebooks\data_analysis.ipynb'

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 1. Data Analysis & Preprocessing\n",
                "\n",
                "This notebook explicitly demonstrates the Exploratory Data Analysis (EDA) and Data Preprocessing (Missing Values, Outliers, Encoding, Scaling) performed on the Health Insurance Cross-Sell Dataset before feeding it into our Machine Learning models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "\n",
                "if os.path.exists('data/raw_data.csv'):\n",
                "    df = pd.read_csv('data/raw_data.csv')\n",
                "elif os.path.exists('../data/raw_data.csv'):\n",
                "    df = pd.read_csv('../data/raw_data.csv')\n",
                "else:\n",
                "    raise FileNotFoundError('Cannot find raw_data.csv')\n",
                "    \n",
                "print(\"Data Shape:\", df.shape)\n",
                "display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.1 Exploratory Data Analysis (EDA) & Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(3, 2, figsize=(16, 18))\n",
                "plt.subplots_adjust(hspace=0.4)\n",
                "\n",
                "# Plot 1: Target Distribution\n",
                "sns.countplot(x='Response', data=df, ax=axes[0, 0], palette='Blues')\n",
                "axes[0, 0].set_title(\"Plot 1: Target Response Breakdown\")\n",
                "\n",
                "# Plot 2: Age Distribution vs Response\n",
                "sns.histplot(data=df, x='Age', hue='Response', bins=30, multiple='stack', ax=axes[0, 1])\n",
                "axes[0, 1].set_title(\"Plot 2: Age Distribution Grouped by Interest\")\n",
                "\n",
                "# Plot 3: Vehicle Damage Impact\n",
                "sns.countplot(x='Vehicle_Damage', hue='Response', data=df, ax=axes[1, 0], palette='rocket')\n",
                "axes[1, 0].set_title(\"Plot 3: Vehicle Damage vs Interest\")\n",
                "\n",
                "# Plot 4: Annual Premium Distribution\n",
                "sns.boxplot(x='Response', y='Annual_Premium', data=df[df['Annual_Premium'] < 100000], ax=axes[1, 1], palette='Set2')\n",
                "axes[1, 1].set_title(\"Plot 4: Annual Premium vs Interest (Filtered Outliers)\")\n",
                "\n",
                "# Plot 5: Correlation Heatmap\n",
                "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
                "sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=\".2f\", ax=axes[2, 0])\n",
                "axes[2, 0].set_title(\"Plot 5: Feature Correlation Heatmap\")\n",
                "\n",
                "axes[2, 1].remove() # Blank out the last subplot space\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.2 Data Preprocessing (Step-by-Step)\n",
                "Data requires heavy preprocessing to align with clustering algorithms mathematically:\n",
                "1. **Missing Value Imputation**: Median for numeric, Mode for categorical.\n",
                "2. **Outlier Treatment**: IQR Clipping applied to `Annual_Premium` due to extreme right skew.\n",
                "3. **Encoding**: Label Encoding for strings like `Gender` and `Vehicle_Damage`.\n",
                "4. **Scaling**: `StandardScaler` to force all variations onto a standard zero-mean variance space."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
                "\n",
                "df_clean = df.copy()\n",
                "if 'id' in df_clean.columns:\n",
                "    df_clean.drop('id', axis=1, inplace=True)\n",
                "\n",
                "# 1. Missing Values\n",
                "for col in ['Age', 'Annual_Premium', 'Vintage']:\n",
                "    df_clean[col].fillna(df_clean[col].median(), inplace=True)\n",
                "for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:\n",
                "    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)\n",
                "print('Missing values handled.')\n",
                "\n",
                "# 2. Outliers (IQR Clipping)\n",
                "Q1 = df_clean['Annual_Premium'].quantile(0.25)\n",
                "Q3 = df_clean['Annual_Premium'].quantile(0.75)\n",
                "IQR = Q3 - Q1\n",
                "lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR\n",
                "df_clean['Annual_Premium'] = df_clean['Annual_Premium'].clip(lower=lower, upper=upper)\n",
                "print('Outliers clipped.')\n",
                "\n",
                "# 3. Encoding\n",
                "vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}\n",
                "df_clean['Vehicle_Age'] = df_clean['Vehicle_Age'].map(vehicle_age_map).fillna(0)\n",
                "le_gender = LabelEncoder()\n",
                "df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])\n",
                "le_damage = LabelEncoder()\n",
                "df_clean['Vehicle_Damage'] = le_damage.fit_transform(df_clean['Vehicle_Damage'])\n",
                "print('Categorical variables encoded.')\n",
                "\n",
                "# Split Response\n",
                "if 'Response' in df_clean.columns:\n",
                "    X_raw = df_clean.drop('Response', axis=1)\n",
                "    y_target = df_clean['Response']\n",
                "else:\n",
                "    X_raw = df_clean\n",
                "\n",
                "# 4. Scaling\n",
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X_raw)\n",
                "print('Features scaled with StandardScaler.')\n",
                "\n",
                "print(\"\\n--- PREPROCESSING COMPLETE ---\")\n",
                "print(\"Final X_scaled shape:\", X_scaled.shape)\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('data_analysis.ipynb successfully overwritten!')
