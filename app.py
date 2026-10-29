import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Machine Learning Classification Dashboard")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

model_option = st.selectbox(
    "Choose Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "svm"
    ]
)

if uploaded_file is not None:
    # 1. Read uploaded CSV
    test_data = pd.read_csv(uploaded_file)
    
    # Clean column names (strip whitespace)
    test_data.columns = test_data.columns.str.strip()
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(test_data.head())

    # 2. Separate target (y) if present
    target_col = "price_range"
    if target_col in test_data.columns:
        y_true = test_data[target_col]
    else:
        y_true = None

    # 3. Clean features (X) by dropping target and identifier/index columns
    cols_to_drop = [col for col in [target_col, "id", "ID", "Unnamed: 0"] if col in test_data.columns]
    X = test_data.drop(columns=cols_to_drop)

    # 4. Load Selected Model
    model_mapping = {
        "logistic_regression": "model/Logistic_Regression.pkl",
        "decision_tree": "model/Decision_Tree.pkl",
        "knn": "model/KNN.pkl",
        "naive_bayes": "model/Naive_Bayes.pkl",
        "svm": "model/SVM.pkl"
    }

    try:
        model_path = model_mapping[model_option]
        model = joblib.load(model_path)
        
        # 5. Generate Predictions
        y_pred = model.predict(X)
        
        # Add predictions to display DataFrame
        results_df = test_data.copy()
        results_df["Predicted_Price_Range"] = y_pred
        
        st.write("### Predictions:")
        st.dataframe(results_df)

        # 6. Display Evaluation Metrics (if target was in original data)
        if y_true is not None:
            st.write("---")
            st.write("### Model Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            mcc = matthews_corrcoef(y_true, y_pred)
            
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Weighted F1 Score", f"{f1:.4f}")
            col3.metric("MCC", f"{mcc:.4f}")

            st.write("#### Confusion Matrix:")
            st.write(confusion_matrix(y_true, y_pred))

    except FileNotFoundError:
        st.error(f"Model file for '{model_option}' not found. Please ensure your models are saved in the `model/` folder.")
    except Exception as e:
        st.error(f"An error occurred while processing predictions: {e}")