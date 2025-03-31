# Diabetes Prediction using SVM

This project implements a Support Vector Machine (SVM) model to predict diabetes using the PIMA Diabetes dataset. The dataset consists of medical diagnostic measurements and aims to classify patients as diabetic or non-diabetic.

## 📂 Dataset
The dataset used in this project is the **PIMA Diabetes Dataset**, which contains the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 = Non-Diabetic, 1 = Diabetic)

## 🛠 Dependencies
Ensure you have the following libraries installed before running the project:
```bash
pip install numpy pandas scikit-learn
```

## 🚀 Project Workflow
### 1️⃣ Importing the Dependencies
The required libraries such as `pandas`, `numpy`, and `sklearn` are imported.

### 2️⃣ Loading and Exploring the Dataset
- The dataset is loaded into a Pandas DataFrame.
- Basic exploratory data analysis (EDA) is performed to understand its structure.
- Statistical summaries and value counts of the target column (`Outcome`) are analyzed.

### 3️⃣ Data Preprocessing
- Features (`X`) and target variable (`Y`) are separated.
- Standardization is applied using `StandardScaler` to normalize the features.

### 4️⃣ Splitting the Dataset
- The dataset is split into **training (80%)** and **testing (20%)** sets using `train_test_split()`.

### 5️⃣ Training the SVM Model
- A **Support Vector Machine (SVM) with a linear kernel** is trained using the training data.

### 6️⃣ Model Evaluation
- The model's accuracy is calculated for both training and testing datasets.

### 7️⃣ Making Predictions
- A **predictive system** is implemented to classify new patient data.
- The input is transformed and classified as diabetic or non-diabetic.

## 📊 Results
The model is evaluated using **accuracy score**:
- **Training Accuracy:** `{training_accuracy}%`
- **Testing Accuracy:** `{testing_accuracy}%`

📌 _Note: Actual accuracy results depend on training execution._

## 📜 Usage
To use this model for prediction, modify the `input_data` in the script:
```python
input_data = (5,166,72,19,175,25.8,0.587,51)
```
Run the script, and the model will predict whether the person is diabetic or not.

## 🏗 Future Enhancements
- Improve model performance by **hyperparameter tuning**.
- Use **different machine learning models** (e.g., Random Forest, Neural Networks).
- Deploy the model using **Flask or Streamlit** for a web-based interface.

## 🤝 Contributing
Feel free to contribute by improving the model, documentation, or adding new features. Fork the repository and create a pull request!

## 📜 License
This project is open-source and available under the **MIT License**.

---
✨ _Happy Coding!_ ✨

