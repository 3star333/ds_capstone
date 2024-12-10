import numpy as np 
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

#define target variables
features = train_data[['Engine Size (L)', 'Year', 'Price (in USD)']]
target_hp = train_data['Horsepower']
target_torque = train_data['Torque (lb-ft)']
target_z60 = train_data['0-60 MPH Time (seconds)']

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


# Evaluate each model


# Predict and calculate error


#estimate 1/4 mile time 
def calculate_quarter_mile_time(horsepower, torque, weight=None):
    """
    Calculates the 1/4 mile time based on horsepower, torque, and optional weight.
    If weight is not provided, uses a default average for sports cars.
    """



#create some graphs here at the bottom 
