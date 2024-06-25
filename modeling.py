# %%


import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.stats import uniform, norm, boxcox
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans, DBSCAN


# Load the CSV file into a DataFrame
vehicles = pd.read_csv('prime_vehicles_df-2.csv')
print(vehicles.info())
print(vehicles.head(10))

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(vehicles[['region', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'state']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['region', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'state']))
# Concatenate the encoded columns with the original dataset
X_df = pd.concat([vehicles, encoded_df], axis=1)
X_df = X_df.drop(columns=['region', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'state'])
#important_features = ['odometer', 'fuel_gas', 'transmission_other', 'fuel_other', 'year', 'price_BoxCox', 'price']
#X_df = X_df[important_features]

print(X_df.info())
print(X_df.head(10))

# Select important features based on permutation importance
important_features = ['odometer', 'year', 'manufacturer_kia', 'manufacturer_volkswagen', 'manufacturer_hyundai']

# Prepare data for modeling
X = X_df[important_features]
y = X_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

print(X_train.shape)
print(X_test.shape)
print(type(X_train), type(y_train))
print(y_train.shape, y_test.shape)

# Define pipelines
pipeline_poly = Pipeline([
    ('poly_features', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('selector', SequentialFeatureSelector(LinearRegression(), n_features_to_select='auto')),
    ('model', LinearRegression())
])

pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

pipeline_lasso = Pipeline([
    ('poly_features', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('selector', SequentialFeatureSelector(Lasso(), n_features_to_select='auto')),
    ('model', Lasso(max_iter=10000))
])

# Define parameter grid
param_grid = [
    {
        'poly_features__degree': [3,4,5],
        'model': [LinearRegression()],
        'model__fit_intercept': [True, False]
    },
]

# Use GridSearchCV to find the best model and parameters
grid_search = GridSearchCV(estimator=pipeline_poly, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)

# Fit GridSearchCV to data
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Model Parameters:", best_params)

# Evaluate on the test set
y_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)

# Predict on the training set
y_train_pred = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)

# Plotting the results for training data
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Train Data')
plt.plot(y_train, y_train, color='red', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted values (Training Data)')
plt.legend()
plt.show()

# Plotting the results for test data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Test Data')
plt.plot(y_test, y_test, color='red', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted values (Test Data)')
plt.legend()
plt.show()

# If you want to convert the GridSearchCV scores to RMSE
results = grid_search.cv_results_
params = results['params']
mean_test_scores = results['mean_test_score']
mean_test_rmse = np.sqrt(-mean_test_scores)

# Print the parameters and their corresponding RMSE scores
for param, score, rmse in zip(params, mean_test_scores, mean_test_rmse):
    print(f"Parameters: {param}, Score (neg_MSE): {score}, RMSE: {rmse}")

'''
# Prepare data for modeling
X = X_df.drop(columns=['price'])
y = X_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(type(X_train), type(y_train))
print(y_train.shape, y_test.shape)

baseline_train = np.ones(shape= y_train.shape)*y_train.mean()
baseline_test = np.ones(shape = y_test.shape)*y_test.mean()
mse_baseline_train = mean_squared_error(baseline_train, y_train)
mse_baseline_test = mean_squared_error(baseline_test, y_test)

print(baseline_train.shape, baseline_test.shape)
print(f'Baseline for training data: {mse_baseline_train}')
print(f'Baseline for testing data: {mse_baseline_test}')

#multicolinearity
 
def vif(exogs, dataframe):
  vif_dict = {}

  for exog in exogs:
    not_exog = [i for i in exogs if i != exog]
    X, y = X_df[not_exog], X_df[exog]

    r_squared = LinearRegression().fit(X,y).score(X,y)

    # calc the VIF
    vif = 1/(1-r_squared)
    vif_dict[exog] = vif

  return pd.DataFrame({"VIF":vif_dict})

#Return table of colinear featrues

vif_data = (vif(X.columns, X).sort_values(by = 'VIF', ascending = False))
vif_data = pd.DataFrame(vif_data)
vif_data.to_csv('vif_df-1.csv', index=False)

#Finding what feature has the highest positive correlation with target

highest_corr = X_df.corr()[['price']].nlargest(columns= 'price', n=2).index[1]
print(highest_corr)


high_vif_features = vif_data[vif_data['VIF'] > 10].index
X_train_reduced = X_train.drop(columns=high_vif_features)
X_test_reduced = X_test.drop(columns=high_vif_features)

# Recalculate VIF for the reduced feature set
vif_data_reduced = vif(X_train_reduced.columns, X_train_reduced).sort_values(by='VIF', ascending=False)
print(vif_data_reduced)

# Retrain model with reduced feature set
model = LinearRegression()
model.fit(X_train_reduced, y_train)

# Evaluate the model performance
y_pred_train = model.predict(X_train_reduced)
y_pred_test = model.predict(X_test_reduced)

mse_train_reduced = mean_squared_error(y_train, y_pred_train)
mse_test_reduced = mean_squared_error(y_test, y_pred_test)

print(f'MSE for training data (reduced): {mse_train_reduced}')
print(f'MSE for testing data (reduced): {mse_test_reduced}')


perm_importance = permutation_importance(model, X_test_reduced, y_test, n_repeats=10, random_state=22)

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'feature': X_test_reduced.columns,
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=False)

print(importance_df)


# List of important features based on permutation importance
important_features = importance_df.head(5)['feature'].tolist()

# Retain only the important features in X_df
X_df_important = X_df[important_features ]
print(X_df_important.info())
print(X_df_important.head(10))

importance_df.to_csv('normalized_cleaned_vehicles_df.csv', index=False)

# Scatter plot for 'odometer' vs 'price'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='odometer', y='price', data=vehicles)
plt.title('Odometer vs Price')
plt.xlabel('Odometer')
plt.ylabel('Price')
plt.show()

# Scatter plot for 'year' vs 'price'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='price', data=vehicles)
plt.title('Year vs Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10,6))
px.scatter(vehicles,x='odometer',y='price',marginal_x="histogram",marginal_y="histogram")
'''