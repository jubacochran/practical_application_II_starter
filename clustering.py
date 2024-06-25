# %%

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
vehicles_normalized = pd.read_csv('normalized_cleaned_vehicles_df.csv')

# Display information and first few rows of the dataset
print(vehicles_normalized.info())
print(vehicles_normalized.head())

# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(vehicles_normalized[['fuel','title_status','transmission','state']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['fuel','title_status','transmission','state']))

# Concatenate the encoded columns with the original dataset
X_df = pd.concat([vehicles_normalized, encoded_df], axis=1)
X_df = X_df.drop(columns=['fuel','title_status','transmission','state'])
important_features = ['odometer', 'fuel_gas', 'transmission_other', 'fuel_other', 'year', 'price_BoxCox','price']
X_df = X_df[important_features]

# Select features for clustering
features = ['odometer', 'year', 'price_BoxCox']
X = vehicles_normalized[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
''''
# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.show()

# Fit K-Means with optimal number of clusters
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# Add cluster labels to the dataframe
vehicles_normalized['cluster_kmeans'] = labels_kmeans
'''
# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Add cluster labels to the dataframe
vehicles_normalized['cluster_dbscan'] = labels_dbscan

# 2D Scatter plot for K-Means clusters
plt.figure(figsize=(20, 10))
sns.scatterplot(x='odometer', y='price_BoxCox', hue='cluster_kmeans', data=vehicles_normalized, palette='viridis')
plt.title('K-Means Clustering')
plt.show()

# 2D Scatter plot for DBSCAN clusters
plt.figure(figsize=(20, 10))
sns.scatterplot(x='odometer', y='price_BoxCox', hue='cluster_dbscan', data=vehicles_normalized, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# 3D Scatter plot for K-Means clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels_kmeans, cmap='viridis')
ax.set_xlabel('Odometer')
ax.set_ylabel('Year')
ax.set_zlabel('Price_BoxCox')
plt.title('3D K-Means Clustering')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()
'''
# 3D Scatter plot for DBSCAN clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels_dbscan, cmap='viridis')
ax.set_xlabel('Odometer')
ax.set_ylabel('Year')
ax.set_zlabel('Price_BoxCox')
plt.title('3D DBSCAN Clustering')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()
'''