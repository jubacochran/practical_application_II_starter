# %%


import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the encoded CSV file into a DataFrame
encoded_vehicles = pd.read_csv('encoded_prime_vehicles.csv')

# Normalize the numerical features
numerical_features = ['price', 'year', 'odometer']
scaler = StandardScaler()
encoded_vehicles[numerical_features] = scaler.fit_transform(encoded_vehicles[numerical_features])

# Define the features and target variable
features = ['year', 'transmission_other', 'condition_good', 'manufacturer_ram', 'manufacturer_audi', 'odometer']
X = encoded_vehicles[features]
y = encoded_vehicles['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline with degree 3
pipeline_poly = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.01))
])

# Perform simple cross-validation by explicitly passing the indices
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_indices = list(kf.split(X_train))

# Fit and evaluate the model using the cross-validation indices
for train_indices, dev_indices in cv_indices:
    X_train_cv, X_dev_cv = X_train.iloc[train_indices], X_train.iloc[dev_indices]
    y_train_cv, y_dev_cv = y_train.iloc[train_indices], y_train.iloc[dev_indices]
    
    pipeline_poly.fit(X_train_cv, y_train_cv)
    y_pred_dev = pipeline_poly.predict(X_dev_cv)
    
    mse = mean_squared_error(y_dev_cv, y_pred_dev)
    mae = mean_absolute_error(y_dev_cv, y_pred_dev)
    rss = np.sum((y_dev_cv - y_pred_dev) ** 2)
    
    print(f"MSE: {mse}, MAE: {mae}, RSS: {rss}")

# Fit the final model on the entire training set
pipeline_poly.fit(X_train, y_train)
y_pred_test = pipeline_poly.predict(X_test)

# Evaluate the model on the test set
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rss = np.sum((y_test - y_pred_test) ** 2)
r2 = r2_score(y_test, y_pred_test)

print(f"Final Model - MSE: {mse}, MAE: {mae}, RSS: {rss}, R-squared: {r2}")

# Plot the residuals to check for homoscedasticity
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot the distribution of residuals to check for normality
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Group the dataset by price ranges: lower, middle, highest
encoded_vehicles['price_group'] = pd.qcut(encoded_vehicles['price'], q=3, labels=['low', 'mid', 'high'])

# Fit a Lasso model to each price group
price_groups = encoded_vehicles.groupby('price_group')

for group_name, group_data in price_groups:
    X_group = group_data[features]
    y_group = group_data['price']
    
    X_train_grp, X_test_grp, y_train_grp, y_test_grp = train_test_split(X_group, y_group, test_size=0.2, random_state=42)
    
    pipeline_poly.fit(X_train_grp, y_train_grp)
    y_pred_grp = pipeline_poly.predict(X_test_grp)
    
    mse_grp = mean_squared_error(y_test_grp, y_pred_grp)
    mae_grp = mean_absolute_error(y_test_grp, y_pred_grp)
    rss_grp = np.sum((y_test_grp - y_pred_grp) ** 2)
    r2_grp = r2_score(y_test_grp, y_pred_grp)
    
    print(f"Price Group: {group_name}")
    print(f"MSE: {mse_grp}, MAE: {mae_grp}, RSS: {rss_grp}, R-squared: {r2_grp}")
    print("\n")
    
    # Plot the residuals for each group
    residuals_grp = y_test_grp - y_pred_grp
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_grp, y=residuals_grp)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {group_name.capitalize()} Price Group')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_grp, kde=True)
    plt.title(f'Residuals Distribution - {group_name.capitalize()} Price Group')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

