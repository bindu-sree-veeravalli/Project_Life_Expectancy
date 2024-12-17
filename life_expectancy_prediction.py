import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'updated_life_expectancy_data.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Display first few rows
df.head()

# Data Preprocessing
df.columns[df.isna().any()].tolist()

pd.isnull(df).sum()

# Check data types and basic statistics
print(df.info())
print(df.describe())

# Fill missing values using median imputation
for col in df.columns[df.isna().any()].tolist():
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Explore distributions of key variables
plt.figure(figsize=(12, 6))
sns.histplot(df['Life expectancy '], kde=True)
plt.title('Distribution of Life Expectancy')
plt.show()

# Analyze relationships between variables
correlation_cols = ['Year', 'Life expectancy ', 'Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling',]

plt.figure(figsize=(10, 10))
sns.heatmap(df[correlation_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Plot histograms for all columns
for col in correlation_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Drop rows with missing values (optional step)
df = df.dropna()  # You can remove this if you want to use imputation instead

# Define independent variables (X) and dependent variables (Y)
X = df.drop(['Life expectancy ', 'Life_expectancy_category'], axis=1)  # Drop target columns
y_continuous = df['Life expectancy ']  # For OLS
y_binary = df['Life_expectancy_category'].apply(lambda x: 1 if x == 'High' else 0)  # For Logit

# Add a constant to the independent variables (for intercept)
X = sm.add_constant(X)

# Convert 'Status' column to 0 and 1 (Label encoding)
X['Status'] = X['Status'].map({'Developed': 1, 'Developing': 0})

# Ensure X and y_continuous are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
X = X.dropna()
y_continuous = y_continuous[X.index]  # Align y_continuous with X

# Fit OLS model
ols_model = sm.OLS(y_continuous, X).fit()

# Print OLS summary
print(ols_model.summary())

# Predict with OLS model
y_pred_ols = ols_model.predict(X)

# Calculate RMSE and R-squared
rmse_ols = np.sqrt(mean_squared_error(y_continuous, y_pred_ols))
r2_ols = r2_score(y_continuous, y_pred_ols)

print(f"OLS RMSE: {rmse_ols}")
print(f"OLS R-squared: {r2_ols}")

# Logit Regression (for binary classification)
logit_model = sm.Logit(y_binary, X).fit()

# Print Logit summary
print(logit_model.summary())

# Predict with Logit model
y_pred_logit = logit_model.predict(X)
y_pred_logit_class = (y_pred_logit >= 0.5).astype(int)  # Convert probabilities to binary classification

# Calculate accuracy
accuracy_logit = accuracy_score(y_binary, y_pred_logit_class)
print(f"Logit Accuracy: {accuracy_logit}")

# Confusion matrix
conf_matrix = confusion_matrix(y_binary, y_pred_logit_class)
print(f"Confusion Matrix:\n{conf_matrix}")

# Precision and Recall for Logit
from sklearn.metrics import precision_score, recall_score

precision_logit = precision_score(y_binary, y_pred_logit_class)
recall_logit = recall_score(y_binary, y_pred_logit_class)

print(f"Logit Precision: {precision_logit}")
print(f"Logit Recall: {recall_logit}")

# Visualizations for OLS

# Scatterplot of "Predicted life expectancy vs Actual values"
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_continuous, y=y_pred_ols, hue=y_continuous, alpha=0.6, palette='viridis')
plt.plot([y_continuous.min(), y_continuous.max()], [y_continuous.min(), y_continuous.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Predicted life expectancy with Actual Values')
plt.legend()
plt.xlim(0, max (max(y_continuous), max(y_pred_ols)))
plt.ylim(0, max (max(y_continuous), max(y_pred_ols)))
plt.show()

# Residual plot for OLS
residuals = y_continuous - y_pred_ols
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_ols, y=residuals, hue=y_continuous, alpha=0.6, palette='plasma')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# ROC Curve for Logit Regression
fpr, tpr, _ = roc_curve(y_binary, y_pred_logit)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix Visualization for Logit
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'], cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Probability Distribution Visualization for Logit
plt.figure(figsize=(10, 6))
sns.histplot(y_pred_logit[y_binary == 1], color='blue', alpha=0.6, label='Actual Positive', kde=True)
sns.histplot(y_pred_logit[y_binary == 0], color='red', alpha=0.6, label='Actual Negative', kde=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Logistic Regression: Predicted Probability Distributions')
plt.legend()
plt.show()

