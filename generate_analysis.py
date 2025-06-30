# generate_analysis.py

def run_analysis():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import joblib
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from xgboost import XGBClassifier

    # Load data
    df = pd.read_csv("startup data.csv")

    # Clean and encode
    df.drop(columns=['Unnamed: 0', 'Unnamed: 6', 'state_code.1', 'object_id', 'id', 'labels',
                     'name', 'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at'],
            inplace=True, errors='ignore')

    for col in ['state_code', 'city', 'category_code']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

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
    df = df[features + ['status']].dropna()
    df['funding_total_usd'] = np.log1p(df['funding_total_usd'])
    df['avg_participants'] = np.log1p(df['avg_participants'])
    df['status'] = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)

    X = df[features]
    y = df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    os.makedirs("static", exist_ok=True)

    # Save metrics to text
    with open("static/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.2%}\n")
        f.write(f"Precision: {prec:.2%}\n")
        f.write(f"Recall: {rec:.2%}\n")
        f.write(f"F1 Score: {f1:.2%}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # Univariate plots
    for col in ['funding_total_usd', 'funding_rounds', 'avg_participants']:
        sns.histplot(df[col], kde=True)
        plt.title(f"Univariate Analysis: {col}")
        plt.savefig(f"static/univariate_{col}.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Multivariate Analysis: Correlation Heatmap")
    plt.savefig("static/multivariate_heatmap.png")
    plt.close()
if __name__ == "__main__":
    run_analysis()
