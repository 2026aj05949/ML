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

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Classification Dashboard")

st.info(
    "Upload a CSV file and select a machine learning model to predict mobile price categories and evaluate performance."
)

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

model_option = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM",
        "Random Forest"
    ]
)

if uploaded_file is not None:

    # Read uploaded CSV
    test_data = pd.read_csv(uploaded_file)

    # Clean column names
    test_data.columns = test_data.columns.str.strip()

    st.subheader("Uploaded Dataset")
    st.dataframe(test_data.head())

    target_col = "price_range"

    if target_col in test_data.columns:
        y_true = test_data[target_col]
    else:
        y_true = None

    cols_to_drop = [
        col for col in [target_col, "id", "ID", "Unnamed: 0"]
        if col in test_data.columns
    ]

    X = test_data.drop(columns=cols_to_drop)

    model_mapping = {
        "Logistic Regression": "model/Logistic_Regression.pkl",
        "Decision Tree": "model/Decision_Tree.pkl",
        "KNN": "model/KNN.pkl",
        "Naive Bayes": "model/Naive_Bayes.pkl",
        "SVM": "model/SVM.pkl",
        "Random Forest": "model/Random_Forest.pkl"
    }

    try:
        model = joblib.load(model_mapping[model_option])

        scaler = joblib.load("model/scaler.pkl")

        # Apply scaling only to models trained on scaled data
        if model_option in [
            "Logistic Regression",
            "KNN",
            "SVM"
        ]:
            X_input = scaler.transform(X)
        else:
            X_input = X

        # Predictions
        y_pred = model.predict(X_input)

        results_df = test_data.copy()
        results_df["Predicted_Price_Range"] = y_pred

        st.subheader("Predictions")
        st.dataframe(results_df)

        # Show metrics if actual labels exist
        if y_true is not None:

            accuracy = accuracy_score(y_true, y_pred)

            precision = precision_score(
                y_true,
                y_pred,
                average="weighted"
            )

            recall = recall_score(
                y_true,
                y_pred,
                average="weighted"
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted"
            )

            mcc = matthews_corrcoef(
                y_true,
                y_pred
            )

            st.subheader("Model Performance")

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col3.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col4.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col5.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            st.dataframe(cm)

    except FileNotFoundError:
        st.error(
            "Model file or scaler file not found."
        )

    except Exception as e:
        st.error(
            f"Error: {str(e)}"
        )

st.markdown("---")
st.caption(
    "Developed by UMESHA M S | AIML Assignment - Mobile Price Classification"
)