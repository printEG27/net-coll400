import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel('allstats.xlsx', sheet_name='Sheet1')

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])
class_names = le.classes_

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

model_accuracy  = accuracy_score(y_pred, y_test)
print("Accuracy:",model_accuracy)

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