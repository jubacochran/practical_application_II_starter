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
columns_to_drop = ['type','size','VIN','paint_color','cylinders']
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

vehicles_numeric = vehicles.select_dtypes(include=['int64', 'float']).drop(columns='id')
print(vehicles_numeric)

vehicles_categorical = vehicles.select_dtypes(include=['object'])
print(vehicles_categorical)

print("=================")
print(vehicles.info())
print(vehicles.describe())
'''

# Initialize the IterativeImputer for numerical data
imputer = IterativeImputer(max_iter=10, random_state=0)

# Create a pipeline with IterativeImputer
pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=0))
])

# Define the parameter grid
param_grid = {
    'imputer__max_iter': [5, 10, 20],
    'imputer__n_nearest_features': [10, 15, 20],
    'imputer__tol': [1e-3, 1e-4, 1e-5]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

with parallel_backend('threading', n_jobs=-1):
    grid_search.fit(vehicles_numeric)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Apply the best parameters to the imputer
best_imputer = IterativeImputer(
    max_iter=5,
    n_nearest_features=10,
    tol=0.001,
    random_state=0
)

# Combine numeric and encoded categorical data for imputation
# Ordinal encode categorical columns
ordinal_cols = ['model', 'fuel', 'title_status', 'transmission', 'state']
ordinal_encoder = OrdinalEncoder()
vehicles_categorical[ordinal_cols] = ordinal_encoder.fit_transform(vehicles_categorical[ordinal_cols])

# Combine numeric and encoded categorical data
combined_data = pd.concat([vehicles_numeric, pd.DataFrame(vehicles_categorical, columns=ordinal_cols)], axis=1)

# Impute the combined data
with parallel_backend('threading', n_jobs=-1):
    combined_data_imputed = best_imputer.fit_transform(combined_data)

# Convert the imputed array back to DataFrame
vehicles_numeric_imputed = pd.DataFrame(combined_data_imputed[:, :vehicles_numeric.shape[1]], columns=vehicles_numeric.columns)
vehicles_categorical_imputed = pd.DataFrame(combined_data_imputed[:, vehicles_numeric.shape[1]:], columns=ordinal_cols)

# Decode ordinal columns back to original categories
vehicles_categorical_imputed[ordinal_cols] = ordinal_encoder.inverse_transform(vehicles_categorical_imputed[ordinal_cols])

# Combine numeric and decoded categorical data
vehicles_imputed = pd.concat([vehicles_numeric_imputed, vehicles_categorical_imputed], axis=1)

# Display the first few rows of the imputed DataFrame
print(vehicles_imputed.head())


output_file_path = 'final_vehicles_df-2.csv'
vehicles_imputed.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")

'''


'''
# Extract ordinal columns for one-hot encoding
ordinal_cols_data = vehicles_imputed[ordinal_cols]

# Create one-hot encoded columns
dummies = pd.get_dummies(ordinal_cols_data, drop_first=True)

# Combine the numeric columns with the one-hot encoded columns
vehicles_numeric_data = vehicles_imputed.drop(columns=ordinal_cols)
final_vehicles_df = pd.concat([vehicles_numeric_data, dummies], axis=1)

# Display the first few rows of the final DataFrame
print(final_vehicles_df.head())

# Display the info of the final DataFrame
print(final_vehicles_df.info())
print(final_vehicles_df.describe())
'''




'''
# Optimize memory usage
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

'''