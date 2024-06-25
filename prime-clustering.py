# %%


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
vehicles = pd.read_csv('optimus-prime-df.csv')

# Recalculate tighter IQR bounds with a stricter multiplier (e.g., 1.0)
Q1_price = vehicles['price'].quantile(0.25)
Q3_price = vehicles['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.0 * IQR_price
upper_bound_price = Q3_price + 1.0 * IQR_price

Q1_odometer = vehicles['odometer'].quantile(0.25)
Q3_odometer = vehicles['odometer'].quantile(0.75)
IQR_odometer = Q3_odometer - Q1_odometer
lower_bound_odometer = Q1_odometer - 1.0 * IQR_odometer
upper_bound_odometer = Q3_odometer + 1.0 * IQR_odometer

# Filter out the outliers with tighter criteria
filtered_vehicles = vehicles[
    (vehicles['price'] >= lower_bound_price) &
    (vehicles['price'] <= upper_bound_price) &
    (vehicles['odometer'] >= lower_bound_odometer) &
    (vehicles['odometer'] <= upper_bound_odometer)
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

# Plotting KMeans Clusters
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features_filtered[:, 0], scaled_features_filtered[:, 1], c=kmeans_labels_filtered, cmap='viridis', alpha=0.5)
plt.title('KMeans Clustering of Price, Odometer, and Year (Tighter Outlier Criteria)')
plt.xlabel('Scaled Price')
plt.ylabel('Scaled Odometer')
plt.colorbar(label='Cluster')
plt.show()

# Add the cluster labels to the DataFrame
filtered_vehicles['kmeans_cluster'] = kmeans_labels_filtered

# Calculate the means of numeric features within each KMeans cluster
numeric_columns = ['price', 'odometer', 'year']
cluster_means = filtered_vehicles.groupby('kmeans_cluster')[numeric_columns].mean()

# Display the cluster means
print(cluster_means)

# Save the cluster means to a CSV file for further inspection
cluster_means.to_csv('kmeans_cluster_means.csv', index=True)
