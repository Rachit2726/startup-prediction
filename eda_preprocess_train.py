# eda_preprocess_train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv("startup data.csv")

# Drop unnecessary columns
df.drop(columns=[
    'Unnamed: 0', 'Unnamed: 6', 'state_code.1', 'object_id', 'id',
    'labels', 'name', 'founded_at', 'closed_at',
    'first_funding_at', 'last_funding_at'
], inplace=True, errors='ignore')

# Encode categorical variables
for col in ['state_code', 'city', 'category_code']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Filter and preprocess
features = [
    'state_code', 'city', 'category_code',
    'age_first_funding_year', 'age_last_funding_year',
    'age_first_milestone_year', 'age_last_milestone_year',
    'relationships', 'funding_rounds', 'funding_total_usd',
    'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate',
    'is_software', 'is_web', 'is_mobile', 'is_enterprise',
    'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech',
    'is_consulting', 'is_othercategory', 'has_VC', 'has_angel',
    'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD',
    'avg_participants', 'is_top500'
]
target = 'status'

df = df[features + [target]].dropna()

# Log transform skewed features
df['funding_total_usd'] = np.log1p(df['funding_total_usd'])
df['avg_participants'] = np.log1p(df['avg_participants'])

# Encode target
df[target] = df[target].apply(lambda x: 1 if x == 'acquired' else 0)

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training with GridSearchCV
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
params = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [3, 5]
}
grid = GridSearchCV(xgb, params, cv=3, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
acc = accuracy_score(y_test, best_model.predict(X_test_scaled))
print(f"Model Accuracy: {acc:.2%}")

# Save model, scaler, and features
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')

print("✅ Model, scaler, and features saved!")
