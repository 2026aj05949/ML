# Mobile Price Classification Using Custom Machine Learning Algorithms

## 📌 Project Overview

This project implements and compares multiple machine learning classification algorithms for predicting the price range of mobile phones based on their specifications. The dataset contains various mobile phone features such as battery power, RAM, screen dimensions, camera specifications, and other technical characteristics.

The objective is to evaluate the performance of different classifiers and identify the most suitable model for mobile price classification.

---

## 🎯 Problem Statement

Given the specifications of a mobile phone, predict its price category (`price_range`).

The target variable consists of four classes:

- **0** → Low Cost
- **1** → Medium Cost
- **2** → High Cost
- **3** → Very High Cost

---

## 📂 Dataset Information

**Dataset:** Mobile Price Classification Dataset

### Features

The dataset contains 20 input features:

- battery_power
- blue
- clock_speed
- dual_sim
- fc
- four_g
- int_memory
- m_dep
- mobile_wt
- n_cores
- pc
- px_height
- px_width
- ram
- sc_h
- sc_w
- talk_time
- three_g
- touch_screen
- wifi

### Target Variable

- price_range

### Dataset Size

- Total Records: 2000
- Features: 20
- Classes: 4

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## 🤖 Custom Machine Learning Models Implemented

The following algorithms were implemented from scratch:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)
6. Random Forest

---

## 📁 Project Structure

```text
ML/
│
├── app.py
├── train_models.py
├── evaluate_models.py
├── train.csv
├── requirements.txt
├── README.md
│
├── CustomLogisticRegression.py
├── CustomDecisionTreeClassifier.py
├── CustomKNeighborsClassifier.py
├── CustomGaussianNB.py
├── CustomSVM.py
├── CustomRandomForestClassifier.py
│
└── model/
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── SVM.pkl
    ├── Random_Forest.pkl
    └── scaler.pkl
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/2026aj05949/ML.git
```

Navigate to the project directory:

```bash
cd ML
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training Models

Train all machine learning models using:

```bash
python train_models.py
```

This will generate and save all trained model files in the `model/` directory.

---

## 📊 Model Evaluation

Run the evaluation script:

```bash
python evaluate_models.py
```

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Area Under Curve (AUC)

---

## 📈 Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | MCC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 66.70% | 70.88% | 66.70% | 62.41% | 0.5884 |
| Decision Tree | 97.25% | 97.25% | 97.25% | 97.25% | 0.9633 |
| KNN | 66.10% | 67.07% | 66.10% | 66.18% | 0.5498 |
| Naive Bayes | 81.70% | 81.78% | 81.70% | 81.73% | 0.7560 |
| SVM | 67.20% | 65.89% | 67.20% | 62.83% | 0.5830 |
| Random Forest | **98.00%** | **98.01%** | **98.00%** | **98.00%** | **0.9733** |

---

## 📌 Observations

### Logistic Regression
- Moderate classification performance.
- Useful as a baseline model.

### Decision Tree
- Very high accuracy.
- Simple and interpretable model.

### KNN
- Sensitive to the choice of neighbors.
- Lower performance compared to tree-based methods.

### Naive Bayes
- Fast training and prediction.
- Good probabilistic classification performance.

### SVM
- Moderate classification accuracy.
- Performance may improve with parameter tuning.

### Random Forest
- Achieved the best results among all models.
- Robust and accurate ensemble learning algorithm.

---

## 🏆 Best Performing Model

### Random Forest Classifier

**Performance Metrics:**

- Accuracy: 98.00%
- Precision: 98.01%
- Recall: 98.00%
- F1 Score: 98.00%
- MCC: 0.9733

Random Forest demonstrated superior performance and is selected as the optimal model for this classification task.

---

## 🌐 Streamlit Application

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

### Features

- Upload dataset
- Select classification model
- Generate predictions
- View evaluation metrics
- Display confusion matrix
- Compare model performance

---

## ✅ Conclusion

This project successfully implemented six machine learning classification algorithms from scratch and evaluated their performance on the Mobile Price Classification dataset.

Among all the models, **Random Forest** achieved the highest accuracy of **98.00%** and the highest MCC score of **0.9733**, making it the most effective model for predicting mobile phone price categories.

The results demonstrate that ensemble learning methods such as Random Forest provide superior predictive performance and generalization capability compared to individual classifiers.

---

## 👨‍🎓 Author

**UMESHA M S**

**Course:** Artificial Intelligence and Machine Learning (AIML)

**GitHub Repository:**
https://github.com/2026aj05949/ML