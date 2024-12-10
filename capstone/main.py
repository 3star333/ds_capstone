import numpy as np 
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def remove_non_numeric_engine_size(data, column_name='Engine Size (L)'):
    # Check if each entry is numeric
    is_numeric = pd.to_numeric(data[column_name], errors='coerce').notna()
    
    # Filter rows where the column is numeric
    cleaned_data = data[is_numeric]
    
    return cleaned_data

#estimate 1/4 mile time 
def calculate_quarter_mile_time(horsepower, torque, weight=None):
    """
    Calculates the 1/4 mile time based on horsepower, torque, and optional weight.
    If weight is not provided, uses a default average for sports cars.
    """

def evaluate_model(model, X_test, y_test, target_name):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print(f"Evaluation for {target_name}:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print()

data = pd.read_csv("capstone\Sport car price.csv")
data['Price (in USD)'] = data['Price (in USD)'].str.replace(',', '').astype(float)

# Convert 'Horsepower' to numeric
data['Horsepower'] = pd.to_numeric(data['Horsepower'], errors='coerce')

# Drop rows with missing or zero values in 'Horsepower' or 'Price (in USD)'
data = data[(data['Horsepower'] > 0) & (data['Price (in USD)'] > 0)]

# Calculate 'Price Per Horsepower'
data['Price Per Horsepower'] = data['Price (in USD)'] / data['Horsepower']

#split data into training sets, 70/30 split 
train_data, test_data = train_test_split(data, test_size=0.3, random_state=925)



data_cleaned = remove_non_numeric_engine_size(data, column_name='Engine Size (L)')

# Reassign features for model preparation
features = data_cleaned[['Engine Size (L)', 'Year', 'Price (in USD)']]
target_hp = data_cleaned['Horsepower']
target_torque = data_cleaned['Torque (lb-ft)']
target_z60 = data_cleaned['0-60 MPH Time (seconds)']

#define target variables
train_features, test_features, train_target_hp, test_target_hp = train_test_split(
    features, target_hp, test_size=0.3, random_state=925
)

train_target_torque, test_target_torque = train_test_split(
    target_torque, test_size=0.3, random_state=925
)

train_target_z60, test_target_z60 = train_test_split(
    target_z60, test_size=0.3, random_state=925
)
# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Horsepower model
model_hp = LinearRegression()
model_hp.fit(X_scaled, target_hp)


# Torque model
model_torque = LinearRegression()
model_torque.fit(X_scaled, target_torque)

# 0-60 MPH Time model
model_z60 = LinearRegression()
model_z60.fit(X_scaled, target_z60)

# Prepare the test data
X_test_scaled = scaler.transform(test_features)

# Evaluate each model
evaluate_model(model_hp, X_test_scaled, test_target_hp, "Horsepower")
evaluate_model(model_torque, X_test_scaled, test_target_torque, "Torque")
evaluate_model(model_z60, X_test_scaled, test_target_z60, "0-60 MPH Time")

# Predict and calculate error






#create some graphs here at the bottom 
