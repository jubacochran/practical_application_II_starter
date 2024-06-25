# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
vehicles = pd.read_csv('vehicles.csv')

# Filter out prices above the threshold and below the lower bound
vehicles = vehicles[(vehicles['price'] <= 45000) & (vehicles['price'] > 5000)]

# Drop columns with a large number of missing values
columns_to_drop = ['type', 'size', 'VIN', 'paint_color', 'cylinders', 'model', 'drive']
vehicles = vehicles.drop(columns=columns_to_drop)

# Apply log transformation and remove outliers for odometer
vehicles['log_odometer'] = np.log1p(vehicles['odometer'])
Q1 = vehicles['log_odometer'].quantile(0.25)
Q3 = vehicles['log_odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
vehicles = vehicles[(vehicles['log_odometer'] >= lower_bound) & (vehicles['log_odometer'] <= upper_bound)]

# Normalize the 'log_odometer' feature
scaler = StandardScaler()
vehicles['scaled_odometer'] = scaler.fit_transform(vehicles[['log_odometer']])

# Define features and target variable
features = ['year', 'scaled_odometer']
X = vehicles[features]
y = vehicles['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'model__alpha': [0.1, 1.0, 10.0]
}

# Define custom cross-validation strategy
cv = [[np.arange(len(X_train)), np.arange(len(X_train), len(X_train) + len(X_test))]]

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)

# Fit GridSearchCV to data
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Plot the residuals to check for homoscedasticity
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot the distribution of residuals to check for normality
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.t
