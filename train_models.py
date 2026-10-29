import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Import your class from the standalone file
from CustomLogisticRegression import CustomLogisticRegression
from CustomDecisionTreeClassifier import CustomDecisionTreeClassifier
from CustomKNeighborsClassifier import CustomKNeighborsClassifier
from CustomGaussianNB import CustomGaussianNB
from CustomSVM import CustomSVM
from CustomRandomForestClassifier import CustomRandomForestClassifier

# 1. Load Data
df = pd.read_csv("train.csv")
X = df.drop(columns=['price_range'])
y = df['price_range']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")
print("Successfully saved scaler.pkl")

# 4. Instantiate Custom Models
custom_models = {
    "Logistic_Regression": CustomLogisticRegression(lr=0.01, num_iter=1000),
    "Decision_Tree": CustomDecisionTreeClassifier(max_depth=10),
    "KNN": CustomKNeighborsClassifier(k=5),
    "Naive_Bayes": CustomGaussianNB(),
    "SVM": CustomSVM(lr=0.001, lambda_param=0.01, n_iters=500),
    "Random_Forest": CustomRandomForestClassifier(n_trees=10, max_depth=10)
}

# 5. Fit & Save
for name, model in custom_models.items():
    if name in ["Logistic_Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
        
    joblib.dump(model, f"model/{name}.pkl")
    print(f"Successfully trained and saved {name}.pkl")