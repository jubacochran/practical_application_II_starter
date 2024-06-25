# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from joblib import parallel_backend

# Load the CSV file into a DataFrame
vehicles = pd.read_csv('vehicles.csv')

# Drop columns with a large number of missing values
columns_to_drop = ['type','size','VIN','paint_color','cylinders','model','drive']
vehicles = vehicles.drop(columns=columns_to_drop)

# Separate numeric and categorical features
numeric_features = vehicles.select_dtypes(include=['int64', 'float']).drop(columns='id')
categorical_features = vehicles.select_dtypes(include=['object']).columns

# Ordinal encode categorical columns
ordinal_encoder = OrdinalEncoder()
encoded_categorical = ordinal_encoder.fit_transform(vehicles[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=categorical_features)

# Combine numeric and encoded categorical data
combined_data = pd.concat([numeric_features.reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

# Define the imputer with the recommended parameters
imputer = IterativeImputer(max_iter=20, n_nearest_features=15, tol=1e-4, random_state=0)

# Perform imputation
with parallel_backend('threading', n_jobs=-1):
    combined_data_imputed = imputer.fit_transform(combined_data)

# Convert the imputed array back to DataFrame
combined_data_imputed_df = pd.DataFrame(combined_data_imputed, columns=combined_data.columns)

# Decode ordinal columns back to original categories
decoded_categorical = ordinal_encoder.inverse_transform(combined_data_imputed_df[categorical_features])
decoded_categorical_df = pd.DataFrame(decoded_categorical, columns=categorical_features)

# Combine numeric and decoded categorical data
vehicles_imputed = pd.concat([combined_data_imputed_df.drop(columns=categorical_features), decoded_categorical_df], axis=1)

# Recalculate tighter IQR bounds with a stricter multiplier (e.g., 1.0) on vehicles_imputed
Q1_price = vehicles_imputed['price'].quantile(0.25)
Q3_price = vehicles_imputed['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.0 * IQR_price
upper_bound_price = Q3_price + 1.0 * IQR_price

Q1_odometer = vehicles_imputed['odometer'].quantile(0.25)
Q3_odometer = vehicles_imputed['odometer'].quantile(0.75)
IQR_odometer = Q3_odometer - Q1_odometer
lower_bound_odometer = Q1_odometer - 1.0 * IQR_odometer
upper_bound_odometer = Q3_odometer + 1.0 * IQR_odometer

# Filter out the outliers with tighter criteria on vehicles_imputed
filtered_vehicles = vehicles_imputed[
    (vehicles_imputed['price'] >= lower_bound_price) &
    (vehicles_imputed['price'] <= upper_bound_price) &
    (vehicles_imputed['odometer'] >= lower_bound_odometer) &
    (vehicles_imputed['odometer'] <= upper_bound_odometer)
]

# Include 'year' as a numerical feature
filtered_vehicles = filtered_vehicles.dropna(subset=['year'])
filtered_vehicles['year'] = filtered_vehicles['year'].astype(int)

# Standardize the 'price', 'odometer', and 'year' features for clustering
scaler = StandardScaler()
scaled_features_filtered = scaler.fit_transform(filtered_vehicles[['price', 'odometer', 'year']])

# Re-apply KMeans Clustering
kmeans_filtered = KMeans(n_clusters=5, random_state=42)
kmeans_labels_filtered = kmeans_filtered.fit_predict(scaled_features_filtered)

# Add the cluster labels to the DataFrame
filtered_vehicles['kmeans_cluster'] = kmeans_labels_filtered

# Save the final dataframe with all data and cluster labels
filtered_vehicles.to_csv('optimus-prime-df.csv', index=False)

print(f"Final dataframe with cluster labels saved to 'optimus-prime-df.csv'")