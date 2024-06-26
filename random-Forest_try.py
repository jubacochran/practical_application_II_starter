# %%



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'optimus-prime-df.csv'  # Change this path to the correct file location
vehicles = pd.read_csv(file_path)

# Drop columns with a large number of missing values if they exist
columns_to_drop = ['type', 'size', 'VIN', 'paint_color', 'cylinders', 'model', 'drive']
columns_to_drop = [col for col in columns_to_drop if col in vehicles.columns]
vehicles = vehicles.drop(columns=columns_to_drop)

# Separate features and target
X = vehicles.drop(columns=['price'])
y = vehicles['price']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Ensure dense output
    ])

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', IterativeImputer(max_iter=20, random_state=0))
])

# Preprocess and impute data
X_preprocessed = pipeline.fit_transform(X)

# Convert the preprocessed data back to DataFrame
preprocessed_columns = numerical_features.tolist() + pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features).tolist()
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=preprocessed_columns)

# Recalculate tighter IQR bounds with a stricter multiplier (e.g., 1.0)
Q1_price = y.quantile(0.25)
Q3_price = y.quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.0 * IQR_price
upper_bound_price = Q3_price + 1.0 * IQR_price

Q1_odometer = vehicles['odometer'].quantile(0.25)
Q3_odometer = vehicles['odometer'].quantile(0.75)
IQR_odometer = Q3_odometer - Q1_odometer
lower_bound_odometer = Q1_odometer - 1.0 * IQR_odometer
upper_bound_odometer = Q3_odometer + 1.0 * IQR_odometer

# Filter out the outliers with tighter criteria
filtered_indices = (y >= lower_bound_price) & (y <= upper_bound_price) & \
                   (vehicles['odometer'] >= lower_bound_odometer) & (vehicles['odometer'] <= upper_bound_odometer)

X_filtered = X_preprocessed_df[filtered_indices]
y_filtered = y[filtered_indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Permutation importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

# Display feature importance
importance_df = pd.DataFrame({
    'feature': X_filtered.columns[sorted_idx],
    'importance': perm_importance.importances_mean[sorted_idx]
})

print(importance_df)

# Select important features (top 10 for instance)
important_features = importance_df['feature'].tail(10).tolist()

# Reduce X to important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Define the models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

# Define the parameter grids for hyperparameter tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0]
    }
}

# Perform grid search with cross-validation for each model
best_models = {}
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    param_grid = {f'regressor__{key}': value for key, value in param_grids[model_name].items()}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_important, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Evaluate each model on the test set
model_results = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test_important)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test)
    r_squared = r2_score(y_test, y_pred)
    model_results[model_name] = (mse, mae, r_squared)

print(model_results)

# Focus on top 10-15% high-priced cars for further analysis
price_threshold = y.quantile(0.85)
high_price_df = vehicles[vehicles['price'] >= price_threshold]
X_high_price = X_preprocessed_df.loc[high_price_df.index, important_features]
y_high_price = high_price_df['price']

# Train the best model on high-priced cars
best_model = best_models['RandomForest']
best_model.fit(X_high_price, y_high_price)
y_pred_high_price = best_model.predict(X_high_price)

# Calculate metrics for high-priced cars
mse_high_price = mean_squared_error(y_high_price, y_pred_high_price)
mae_high_price = mean_absolute_error(y_high_price, y_pred_high_price)
r_squared_high_price = r2_score(y_high_price, y_pred_high_price)

print("Metrics for the best RandomForest model on high-priced cars:")
print(f"MSE: {mse_high_price}")
print(f"MAE: {mae_high_price}")
print(f"R-squared: {r_squared_high_price}")

# Plot feature importance for high-priced cars
importances = best_model.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]
features = X_high_price.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest) on High Priced Cars")
plt.bar(range(X_high_price.shape[1]), importances[indices], align='center')
plt.xticks(range(X_high_price.shape[1]), features[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()