# main.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Load dataset
def load_data():
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    return X, y, feature_names

# Train and evaluate models
def train_evaluate_models(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    print("Decision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
    joblib.dump(dt_model, 'models/decision_tree.pkl')

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
    joblib.dump(rf_model, 'models/random_forest.pkl')

    # Support Vector Machine
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
    joblib.dump(svm_model, 'models/svm.pkl')

if __name__ == "__main__":
    X, y, feature_names = load_data()
    train_evaluate_models(X, y)

