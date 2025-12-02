import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings


print("--- PHASE 1: DATA EXPLORATION ---")

# 1. Read CSV
filename = 'mobile phone price prediction.csv'
data = pd.read_csv(filename)

# 2. Print Top 5 rows
print("\nTop 5 rows:")
print(data.head())

# 3. Print Bottom 5 rows
print("\nBottom 5 rows:")
print(data.tail())

# 4. Print Shape (Rows and Columns)
print(f"\nRows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")

# 5. Check for Null Values
print("\nNull Values per column (Before Cleaning):")
print(data.isnull().sum())

# 6. Check Data Types
print("\nColumn DataTypes:")
print(data.dtypes)

print("\n--- PHASE 2: PRE-PROCESSING ---")

if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

for col in data.columns:
    if data[col].dtype == 'object':
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna("Unknown")
    else:
        data[col] = data[col].fillna(data[col].mean())

print("Null values filled.")

if data['Price'].dtype == 'object':
    data['Price'] = data['Price'].str.replace(',', '')
    data['Price'] = data['Price'].astype(np.int64)
    print("'Price' column converted to Integer.")

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

print(f"Feature Set (X) Shape: {X.shape}")
print(f"Target Label (Y) Shape: {Y.shape}")
cat_columns = X.select_dtypes(['object']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])

if (X < 0).any().any():
    X[X < 0] = 0
    print("Negative values handled for MultinomialNB compatibility.")

print("Data Pre-processing Complete.")

print("\n--- PHASE 3: CLASSIFICATION ---")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
classifiers = {
    'Bernoulli': BernoulliNB(),
    'Random Forest': RandomForestClassifier(),
    'Gaussian': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Multinomial': MultinomialNB(),
    'KNeighbors': KNeighborsClassifier()
}

# Lists to store metrics for plotting
names = []
acc_scores = []
prec_scores = []
recall_scores = []
f1_scores = []

print(f"{'Classifier':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 75)

# 2. Loop through all classifiers and evaluate
for name, model in classifiers.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    # 'weighted' is used because we have multiple target classes (processor names)
    acc = metrics.accuracy_score(Y_test, y_pred)
    prec = metrics.precision_score(Y_test, y_pred, average='weighted', zero_division=0)
    rec = metrics.recall_score(Y_test, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(Y_test, y_pred, average='weighted', zero_division=0)
    
    # Store for plotting
    names.append(name)
    acc_scores.append(acc)
    prec_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    
    print(f"{name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

print("\n--- PHASE 4: VISUALIZATION ---")

# Graph 1: Line Plot (Recall vs F1 Score)
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(names))

plt.plot(x_axis, recall_scores, marker='o', label='Recall', color='blue', linewidth=2)
plt.plot(x_axis, f1_scores, marker='s', label='F1 Score', color='orange', linewidth=2)

plt.xticks(x_axis, names)
plt.title("Comparison of Classifiers: Recall vs F1 Score")
plt.xlabel("Classifier Model")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

print("Displaying Line Graph... (Close the plot window to proceed)")
plt.show()

# Graph 2: Bar Plot (F1 Scores)
plt.figure(figsize=(14, 7))
colors = ['#08737f', '#44a1a0', '#7fd3c9', '#b9e6e0', '#f0f9f8', '#004c4c']

bars = plt.bar(names, f1_scores, width=0.7, color=colors, edgecolor='black')

plt.title("F1 Score Performance by Classifier")
plt.xlabel("Classifier Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1.1) # Set y-limit slightly above 1 for clarity

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

print("Displaying Bar Graph...")
plt.show()

print("\nProject Tasks Completed Successfully.")