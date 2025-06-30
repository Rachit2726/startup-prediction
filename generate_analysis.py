# generate_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load pre-trained components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Load dataset
df = pd.read_csv("startup data.csv")

# Clean data
df.drop(columns=[
    'Unnamed: 0', 'Unnamed: 6', 'state_code.1', 'object_id', 'id',
    'labels', 'name', 'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'
], inplace=True, errors='ignore')

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
for col in ['state_code', 'city', 'category_code']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Filter valid rows
df = df[features + ['status']].dropna()
df['funding_total_usd'] = np.log1p(df['funding_total_usd'])
df['avg_participants'] = np.log1p(df['avg_participants'])
df['status'] = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)

# Train-test split
X = df[features]
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Create static folder
os.makedirs("static", exist_ok=True)

# Save metrics
with open("static/metrics.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(f"Precision: {report['1']['precision']:.2f}\n")
    f.write(f"Recall: {report['1']['recall']:.2f}\n")
    f.write(f"F1-Score: {report['1']['f1-score']:.2f}\n")
    f.write(f"Accuracy: {report['accuracy']:.2f}\n")

# Save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

# Univariate Analysis
for col in ['funding_total_usd', 'funding_rounds', 'avg_participants']:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='orange')
    plt.title(f"Univariate Analysis: {col}")
    plt.tight_layout()
    plt.savefig(f"static/univariate_{col}.png")
    plt.close()

# Multivariate Analysis (Correlation heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Multivariate Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/multivariate_heatmap.png")
plt.close()

print("✅ Analysis saved in 'static/' folder")
