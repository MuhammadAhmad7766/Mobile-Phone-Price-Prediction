# model_build.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Changed to Regressor for Price
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
data = pd.read_csv('mobile phone price prediction.csv')

# 2. Data Cleaning
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Drop 'Name' (Too unique)
if 'Name' in data.columns:
    data.drop('Name', axis=1, inplace=True)

# Fill Nulls
for col in data.columns:
    if data[col].dtype == 'object':
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna("Unknown")
    else:
        data[col] = data[col].fillna(data[col].mean())

# Fix Price Column (Remove commas)
if data['Price'].dtype == 'object':
    data['Price'] = data['Price'].str.replace(',', '')
    data['Price'] = data['Price'].astype(np.int64)

# 3. Separate Features (X) and Target (Y)
# AB TARGET 'Price' HAI
Y = data['Price'] 
X = data.drop('Price', axis=1) # Drop Price from features

# 4. Encoding Categorical Data
encoders = {}
cat_columns = X.select_dtypes(['object']).columns

for col in cat_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Note: Y (Price) is already a number, so no LabelEncoding needed for Y.

# 5. Train Model (Using Regressor)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Using Random Forest Regressor for better accuracy on prices
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)
print(f"Model Accuracy (R2 Score): {score:.2f}")

# 6. Save Model and Encoders
data_to_save = {
    'model': model,
    'encoders': encoders,
    'feature_names': X.columns.tolist()
}

with open('mobile_price_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Price Prediction Model saved as 'mobile_price_model.pkl'")