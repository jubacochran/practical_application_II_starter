# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, norm
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
import plotly.express as px

# Load the CSV file into a DataFrame
vehicles = pd.read_csv('vehicles.csv')

# Calculate missing values in each column
missing_values = vehicles.isnull().sum().plot(kind='bar')

# Set a threshold based on the specified value
threshold = 45000

# Filter out prices above the threshold
vehicles = vehicles[vehicles['price'] <= threshold]

# Display the first few rows of the cleaned dataframe
print(vehicles.head(10))
print(vehicles.info())

# Filter out prices above the threshold
filtered_prices = vehicles['price'].dropna()
vehicles = vehicles[(vehicles['price'] <= threshold) & (vehicles['price'] > 5000)]

# Define a threshold for dropping columns missing more than 50% of values.
#threshold = 0.5 * len(vehicles)
#columns_to_drop = missing_values[missing_values > threshold].index

# Cleaning DataFrame
vehicles['region'] = vehicles['region'].str.replace(r' / ', ' - ', regex=False)
vehicles = vehicles.dropna(subset=['year'])
vehicles['year'] = vehicles['year'].astype('int64')

# Drop columns with a large number of missing values
columns_to_drop = ['type','size','VIN','paint_color','cylinders','model','drive']
vehicles = vehicles.drop(columns=columns_to_drop)

print(vehicles.info())

# Plot the distribution of 'price' after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(vehicles['price'], bins=30, kde=True)
plt.title('Price Distribution After Removing Outliers')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Calculate IQR
Q1 = vehicles['price'].quantile(0.25)
Q3 = vehicles['price'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers

vehicles = vehicles[(vehicles['price'] >= lower_bound) & (vehicles['price'] <= upper_bound)]

# Filter out outliers for 'odometer'
vehicles = vehicles[(vehicles['odometer'] >= lower_bound) & (vehicles['odometer'] <= upper_bound)]

# Set a lower limit for odometer values (e.g., 1,000 miles)
lower_limit_odometer = 1000
vehicles = vehicles[vehicles['odometer'] >= lower_limit_odometer]

# Calculate IQR for 'odometer'
Q1 = vehicles['odometer'].quantile(0.25)
Q3 = vehicles['odometer'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for 'odometer'
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers for 'odometer'
vehicles = vehicles[(vehicles['odometer'] >= lower_bound) & (vehicles['odometer'] <= upper_bound)]

# Plot the distribution of 'odometer' after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(vehicles['odometer'], bins=30, kde=True)
plt.title('Odometer Distribution After Removing Outliers')
plt.xlabel('Odometer')
plt.ylabel('Frequency')
plt.show()

odometer_skewness_no_outliers = vehicles['odometer'].skew()
print(f"Skewness of odometer after removing outliers: {odometer_skewness_no_outliers}")

# Plot the distribution of 'price' after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(vehicles['price'], bins=30, kde=True)
plt.title('Price Distribution After Removing Outliers')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Calculate skewness after removing outliers
price_skewness_no_outliers = vehicles['price'].skew()
print(f"Skewness of price after removing outliers: {price_skewness_no_outliers}")
print(vehicles.head(20))
print(vehicles.info())
print(vehicles.shape)

missing_values2 = vehicles.isnull().sum().plot(kind='bar')

numeric_features = vehicles.select_dtypes(include=['int64', 'float']).drop(columns='id')
print(numeric_features)

categorical_features = vehicles.select_dtypes(include=['object'])
print(categorical_features)

print("=================")
print(vehicles.info())
print(vehicles.describe())

# Separate numeric and categorical features
numeric_features = vehicles.select_dtypes(include=['int64', 'float']).drop(columns='id')
categorical_features = vehicles.select_dtypes(include=['object']).columns

print(numeric_features)
print(categorical_features)

print("=================")
print(vehicles.info())
print(vehicles.describe())

# Ordinal encode categorical columns
ordinal_encoder = OrdinalEncoder()
encoded_categorical = ordinal_encoder.fit_transform(vehicles[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=categorical_features)

# Combine numeric and encoded categorical data
combined_data = pd.concat([numeric_features.reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

# Define the pipeline for imputation
pipeline = Pipeline(steps=[
    ('imputer', IterativeImputer())
])

# Define the parameter grid
param_grid = {
    'imputer__max_iter': [5, 10, 20],
    'imputer__n_nearest_features': [10, 15, 20],
    'imputer__tol': [1e-3, 1e-4, 1e-5]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Perform imputation with parallel backend
with parallel_backend('threading', n_jobs=-1):
    best_imputer = grid_search.fit(combined_data)

# Impute the combined data
with parallel_backend('threading', n_jobs=-1):
    combined_data_imputed = best_imputer.transform(combined_data)

# Convert the imputed array back to DataFrame
combined_data_imputed_df = pd.DataFrame(combined_data_imputed, columns=combined_data.columns)

# Decode ordinal columns back to original categories
decoded_categorical = ordinal_encoder.inverse_transform(combined_data_imputed_df[categorical_features])
decoded_categorical_df = pd.DataFrame(decoded_categorical, columns=categorical_features)

# Combine numeric and decoded categorical data
vehicles_imputed = pd.concat([combined_data_imputed_df.drop(columns=categorical_features), decoded_categorical_df], axis=1)
print(vehicles_imputed.info())
print(vehicles_imputed.head(10))
# Save the imputed DataFrame to a CSV file
output_file_path = 'prime_vehicles_df-2.csv'
vehicles_imputed.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")