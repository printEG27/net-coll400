import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load data from Excel
df = pd.read_excel('allstats.xlsx', sheet_name='Sheet1')

# Encode website labels as numbers
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])
class_names = le.classes_

# Train/test split
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Normalize features
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0) + 1e-8

X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

# Fit logistic regression on training set
regression = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
)
regression.fit(X_train, y_train)

# Predict on test set
y_pred = regression.predict(X_test)

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
ax.set_title('Logistic Regression Multi-class Classification')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('logistic_regression_plot.png', dpi=150)
plt.show()
print("Plot saved #awesome")