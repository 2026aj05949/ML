import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

# Import custom classes so joblib can unpickle saved model instances
try:
    from CustomDecisionTreeClassifier import CustomDecisionTreeClassifier, Node
    from CustomRandomForestClassifier import CustomRandomForestClassifier
    from CustomLogisticRegression import CustomLogisticRegression
    from CustomKNeighborsClassifier import CustomKNeighborsClassifier
    from CustomGaussianNB import CustomGaussianNB
    from CustomSVM import CustomSVM
except ImportError:
    pass


# 1. Load Dataset
data_path = "train.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset standard path '{data_path}' not found.")

data = pd.read_csv(data_path)

# Features and Target
X = data.drop("price_range", axis=1)
y = data["price_range"].values

# 2. Load Scaler
scaler_path = "model/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# 3. Models Dictionary
models = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "SVM": "model/SVM.pkl",
    "Random Forest": "model/Random_Forest.pkl",
}

results = []

print("=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

for model_name, model_path in models.items():

    if not os.path.exists(model_path):
        print(f"\n[SKIP] Model file '{model_path}' not found.")
        continue

    # Load model
    model = joblib.load(model_path)

    # Scale data for distance-based / linear models
    if model_name in ["Logistic Regression", "KNN", "SVM"] and scaler:
        X_input = scaler.transform(X)
    else:
        X_input = X.values if hasattr(X, "values") else X

    # Ensure 2D numpy array format
    X_input = np.asarray(X_input)
    if X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)

    # Predictions
    y_pred = model.predict(X_input)

    # Calculate Probabilities for ROC-AUC
    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_input)
            y_prob = np.asarray(y_prob)
            unique_classes = np.unique(y)

            if len(unique_classes) == 2:
                # Binary classification check
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_auc = y_prob[:, 1]
                else:
                    y_prob_auc = y_prob
                auc = roc_auc_score(y, y_prob_auc)
            else:
                # Multi-class classification check
                auc = roc_auc_score(y, y_prob, multi_class="ovr")
        except Exception as e:
            print(f"Warning: Could not compute AUC for {model_name}: {e}")
            auc = np.nan
    else:
        auc = np.nan

    # Compute Standard Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    results.append([
        model_name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    print(f"\nModel : {model_name}")
    print("-" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}" if not np.isnan(auc) else "AUC       : N/A")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

# 4. Create & Save Results DataFrame
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

print("\n")
print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

# Save Results to CSV
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nResults saved to 'model_comparison_results.csv'")