# Mobile Price Classification Using Custom Machine Learning Algorithms

## 📌 Live Demo

### GitHub Repository
https://github.com/2026aj05949/ML

### Streamlit Application
https://2026aj05949.streamlit.app/

---

## 📖 Project Overview

This project implements and compares multiple machine learning classification algorithms for predicting the price range of mobile phones based on their technical specifications. The goal is to classify mobile phones into different price categories and compare the performance of various custom-built machine learning models.

The project includes model training, evaluation, comparison, and deployment using Streamlit.

---

## 🎯 Problem Statement

The objective of this project is to predict the price category of a mobile phone using machine learning algorithms based on features such as battery power, RAM, screen resolution, camera specifications, and other hardware characteristics.

### Price Categories

- **0** → Low Cost
- **1** → Medium Cost
- **2** → High Cost
- **3** → Very High Cost

---

## 📊 Dataset Information

### Dataset Name

Mobile Price Classification Dataset

### Dataset Details

- Total Records: **2000**
- Input Features: **20**
- Target Classes: **4**
- Target Variable: **price_range**

### Features

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

The following machine learning algorithms were implemented from scratch:

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
├── requirements.txt
├── README.md
├── train.csv
│
├── CustomLogisticRegression.py
├── CustomDecisionTreeClassifier.py
├── CustomKNeighborsClassifier.py
├── CustomGaussianNB.py
├── CustomSVM.py
├── CustomRandomForestClassifier.py
│
├── model_comparison_results.csv
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

Move into the project directory:

```bash
cd ML
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Models

Run the training script:

```bash
python train_models.py
```

This script:

- Loads the dataset
- Performs train-test split
- Applies feature scaling
- Trains all custom machine learning models
- Saves trained models into the `model/` directory

---

## 📈 Evaluate the Models

Run:

```bash
python evaluate_models.py
```

### Evaluation Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | MCC |
|---------|---------:|---------:|---------:|---------:|---------:|
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
- Achieved excellent performance.
- Easy to understand and interpret.

### KNN
- Sensitive to neighboring instances.
- Lower performance compared to tree-based methods.

### Naive Bayes
- Fast and efficient probabilistic classifier.
- Delivered good performance.

### SVM
- Moderate classification performance.
- May benefit from additional tuning.

### Random Forest
- Achieved the highest performance among all models.
- Most robust and accurate model.

---

## 🏆 Best Performing Model

### Random Forest Classifier

**Performance Metrics**

- Accuracy: **98.00%**
- Precision: **98.01%**
- Recall: **98.00%**
- F1 Score: **98.00%**
- MCC: **0.9733**

Random Forest outperformed all other classifiers and was selected as the final model for mobile price classification.

---

## 🌐 Streamlit Application

Launch the dashboard:

```bash
streamlit run app.py
```

### Features

✅ Upload CSV datasets

✅ Select machine learning model

✅ Generate predictions

✅ Display uploaded dataset

✅ View prediction results

✅ Display Accuracy, Precision, Recall, F1 Score, and MCC

✅ View Confusion Matrix

✅ User-friendly interface

---

## 📸 Application Screenshots

### Dashboard Home

Add screenshot:

```text
screenshots/dashboard.png
```

### Prediction Results

Add screenshot:

```text
screenshots/predictions.png
```

---

## ✅ Final Submission Checklist

- [x] GitHub repository link works
- [x] Streamlit app link opens correctly
- [x] App loads without errors
- [x] All required features implemented
- [x] Custom machine learning models implemented
- [x] Model evaluation completed
- [x] Performance comparison generated
- [x] README.md completed
- [x] Streamlit application deployed

---

## 🎯 Conclusion

This project successfully implemented six machine learning classification algorithms from scratch and evaluated their performance on the Mobile Price Classification dataset.

The comparative analysis showed that ensemble learning methods significantly outperform individual classifiers. Among all tested models, **Random Forest** achieved the highest accuracy of **98.00%** and the highest MCC score of **0.9733**, making it the most suitable model for predicting mobile phone price categories.

The project includes model training, evaluation, comparison, and deployment through an interactive Streamlit web application.

---

## 👨‍🎓 Author

**UMESHA M S**

Artificial Intelligence and Machine Learning (AIML)

### Repository Link

https://github.com/2026aj05949/ML

### Streamlit Deployment

https://2026aj05949.streamlit.app/