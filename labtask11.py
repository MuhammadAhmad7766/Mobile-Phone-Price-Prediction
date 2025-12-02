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
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Preparation (Required for Lab 11)
# ==========================================
print("Loading and preparing data...")

# Load Data
filename = 'mobile phone price prediction.csv'
data = pd.read_csv(filename)

# Drop index column if present
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Fill Null Values
for col in data.columns:
    if data[col].dtype == 'object':
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna("Unknown")
    else:
        data[col] = data[col].fillna(data[col].mean())

# Fix 'Price' column (remove commas and convert to int)
if data['Price'].dtype == 'object':
    data['Price'] = data['Price'].str.replace(',', '')
    data['Price'] = data['Price'].astype(np.int64)

# Split Features (X) and Target (Y)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

# Convert Object columns to Integers (Factorization)
cat_columns = X.select_dtypes(['object']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])

# Fix Negative Values (created by missing data during factorization) for MultinomialNB
if (X < 0).any().any():
    X[X < 0] = 0

# Split into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

print("Data ready.")

# ==========================================
# 2. Classifier Execution
# ==========================================
print("\nRunning Classifiers...")

classifiers_names = ['Bernoulli', 'Random Forest', 'Gaussian', 'Decision Tree', 'Multinomial', 'KNeighbors']
# Lists to store metrics
recall_scores = []
f1_scores = []

# Helper function to fit model and get scores
def get_scores(model, name):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # Calculate scores (weighted average for multi-class)
    rec = metrics.recall_score(Y_test, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(Y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"{name} -> Recall: {rec:.4f}, F1: {f1:.4f}")
    return rec, f1

# Run all models
r, f = get_scores(BernoulliNB(), 'Bernoulli')
recall_scores.append(r); f1_scores.append(f)

r, f = get_scores(RandomForestClassifier(), 'Random Forest')
recall_scores.append(r); f1_scores.append(f)

r, f = get_scores(GaussianNB(), 'Gaussian')
recall_scores.append(r); f1_scores.append(f)

r, f = get_scores(DecisionTreeClassifier(), 'Decision Tree')
recall_scores.append(r); f1_scores.append(f)

r, f = get_scores(MultinomialNB(), 'Multinomial')
recall_scores.append(r); f1_scores.append(f)

r, f = get_scores(KNeighborsClassifier(), 'KNeighbors')
recall_scores.append(r); f1_scores.append(f)

# ==========================================
# 3. Plotting Graphs
# ==========================================
print("\nGenerating Line Plot...")

# Plot 1: Line Graph (Recall vs F1)
plt.figure(figsize=(10, 6))
x_axis = np.arange(len(classifiers_names))

plt.plot(x_axis, recall_scores, marker='o', label='Recall', color='blue')
plt.plot(x_axis, f1_scores, marker='o', label='F1', color='orange')

plt.xticks(x_axis, classifiers_names) # Set text labels for x-axis
plt.title("Scores of Applied Classifiers (Recall vs F1)")
plt.xlabel("Classifier")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

# Save figure just in case show() doesn't work
plt.savefig("lab11_line_graph.png") 
print("Line graph saved to 'lab11_line_graph.png'. Now displaying...")
plt.show() # NOTE: Close this window to see the next graph

# Plot 2: Bar Graph (F1 Scores)
print("\nGenerating Bar Plot...")
plt.figure(figsize=(12, 8))
left = np.arange(len(classifiers_names))
height = f1_scores
tick_label = classifiers_names
colors = ['#08737f', '#44a1a0', '#7fd3c9', '#b9e6e0', '#f0f9f8', '#004c4c']

plt.bar(left, height, tick_label=tick_label, width=0.8, color=colors)
plt.title("F1 Scores of All Applied Classifiers")
plt.xlabel("Classifier")
plt.ylabel("F1 Score")

# Save figure
plt.savefig("lab11_bar_graph.png")
print("Bar graph saved to 'lab11_bar_graph.png'. Now displaying...")
plt.show()

print("\nDone.")