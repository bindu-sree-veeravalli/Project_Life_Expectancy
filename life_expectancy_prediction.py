# -*- coding: utf-8 -*-
"""
Life Expectancy Prediction Analysis

This script performs data preprocessing, exploratory data analysis (EDA),
OLS regression, and logistic regression using the Life Expectancy dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "updated_life_expectancy_data.csv"  # Replace with your dataset filename
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset not found. Ensure 'updated_life_expectancy_data.csv' is in the same directory.")
    exit()

# Handle missing values using median imputation
for col in df.columns[df.isna().any()]:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Convert 'Status' column to binary encoding (Developed=1, Developing=0)
if 'Status' in df.columns:
    df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})

# Define independent and dependent variables
X = df.drop(['Life expectancy '], axis=1, errors='ignore')  # Drop target column
y_continuous = df['Life expectancy ']  # For OLS
y_binary = (y_continuous >= y_continuous.median()).astype(int)  # For Logit (binary classification)

# Add a constant for regression models
X = sm.add_constant(X)

# --- OLS Regression ---
print("\n--- OLS Regression ---")
ols_model = sm.OLS(y_continuous, X).fit()
print(ols_model.summary())

# Calculate OLS metrics
y_pred_ols = ols_model.predict(X)
rmse_ols = np.sqrt(mean_squared_error(y_continuous, y_pred_ols))
r2_ols = r2_score(y_continuous, y_pred_ols)
print(f"OLS RMSE: {rmse_ols:.2f}, R-squared: {r2_ols:.2f}")

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
logit_model = sm.Logit(y_binary, X).fit()
print(logit_model.summary())

# Calculate Logistic Regression metrics
y_pred_logit = logit_model.predict(X)
y_pred_class = (y_pred_logit >= 0.5).astype(int)
accuracy_logit = accuracy_score(y_binary, y_pred_class)
conf_matrix = confusion_matrix(y_binary, y_pred_class)
print(f"Logit Accuracy: {accuracy_logit:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# --- Visualizations ---
def plot_ols_results(y_true, y_pred):
    """Scatter plot of actual vs predicted values and residuals."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Fit')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title('OLS: Actual vs Predicted')
    plt.legend()
    plt.show()

    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='green')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('OLS Residuals')
    plt.show()

def plot_logit_roc(y_true, y_pred_prob):
    """ROC curve for logistic regression."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression: ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_ols_results(y_continuous, y_pred_ols)
plot_logit_roc(y_binary, y_pred_logit)

# Save processed dataset (optional)
df.to_csv("processed_life_expectancy_data.csv", index=False)
print("Processed dataset saved as 'processed_life_expectancy_data.csv'.")
