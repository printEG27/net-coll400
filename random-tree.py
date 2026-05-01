import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Import the data
df = pd.read_excel('allstats.xlsx', sheet_name='Sheet1')

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])
class_names = le.classes_

feature_cols = ['Packet count', 
                'Total packet length', 
                'Average packet length',
                'Minimum packet length', 
                'Maximum packet length', 
                'Packet length variance', 
                'Avg. packet interval', 
                'Min. packet interval', 
                'Max. packet interval',
                'Packet itvl variance'
                ]

# normalize the features
X = df[feature_cols].values
preprocessing.normalize(X)

y = df['label_encoded'].values

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.5, 
    random_state=42, 
    stratify=y)

# Initialize classifier
classifier = RandomForestClassifier(
    max_depth=1000,
    n_estimators=100, 
    random_state=42)
classifier.fit(X_train, y_train)

# Get feature importances
importances = classifier.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# predict!
y_pred = classifier.predict(X_test)

model_accuracy  = accuracy_score(y_pred, y_test)
print("Accuracy:",model_accuracy)

# Make the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.savefig('random_tree_confusion_matrix.png', dpi=150)
plt.show()

# Plot predicted vs actual and show 
fig, ax = plt.subplots(figsize=(10, 5))

x_axis = np.arange(len(y_test))

ax.plot(x_axis, y_pred, color='steelblue', linewidth=1.5, label='prediction')
ax.plot(x_axis, y_test, color='crimson',   linewidth=1.5, linestyle='--', label='real_values')

ax.set_yticks(np.arange(len(class_names)))
ax.set_yticklabels(class_names)
ax.set_xlabel('Index')
ax.set_title('Random Tree Multi-class Classification')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('random_tree_plot.png', dpi=150)
plt.show()

