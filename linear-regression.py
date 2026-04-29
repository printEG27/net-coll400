import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score

# Load data from Excel
df = pd.read_excel('allstats.xlsx', sheet_name='Sheet1')

# Encode website labels as numbers
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])
class_names = le.classes_

# Train/test split (60/20/20 like the example report he gave us)
n = len(df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

feature_cols = ['Packet count', 
                'Total packet length', 
                'Average packet length',
                'Minimum packet length', 
                'Maximum packet length', 
                'Most common packet length', 
                'Avg. packet interval', 
                'Min. packet interval', 
                'Max. packet interval']
X = df[feature_cols].values
y = df['label_encoded'].values

# Shuffle before splitting
np.random.seed(42)
shuffle_idx = np.random.permutation(n)
X, y = X[shuffle_idx], y[shuffle_idx]

train_end = int(0.6 * n)
X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[train_end:], y[train_end:]

# Fit linear regression on training set
slr = LinearRegression()
slr.fit(X_train, y_train)

# Predict on test set
y_pred_raw = slr.predict(X_test)
y_pred = np.clip(np.round(y_pred_raw).astype(int), 0, len(class_names) - 1)

# Calculate accuracy
model_accuracy  = accuracy_score(y_pred,y_test)
print("Accuracy:",model_accuracy)

# Make plot
fig, ax = plt.subplots(figsize=(10, 5))

x_axis = np.arange(len(y_test))

ax.plot(x_axis, y_pred, color='steelblue', linewidth=1.5, label='prediction')
ax.plot(x_axis, y_test, color='crimson',   linewidth=1.5, linestyle='--', label='real_values')

ax.set_yticks(np.arange(len(class_names)))
ax.set_yticklabels(class_names)
ax.set_xlabel('Index')
ax.set_title('Linear Regression Multi-class Classification')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=150)
plt.show()
print("Plot saved #awesome")