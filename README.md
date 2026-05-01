# 📊 Student Productivity Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview

This project is a **Machine Learning-based Student Productivity Prediction System** that predicts a student's productivity score (out of 100) using academic habits, lifestyle factors, and digital distractions.

It uses multiple models and combines them using a **weighted ensemble approach** to provide a reliable final prediction.

---

## 🎯 Objectives

- Predict student productivity using real-world features  
- Compare multiple ML models  
- Improve prediction reliability using weighted averaging  
- Build an interactive dashboard  

---

## 🧠 Models Used

- Decision Tree  
- Multilinear Regression  
- Polynomial Regression  
- Artificial Neural Network (ANN)  

---

## 📂 Dataset

- Source: Kaggle  
- Name: *Student Productivity and Digital Distraction Dataset*  
- Records: ~20,000  
- Features: 18  

### 🔑 Key Features

- Study Hours Per Day  
- Focus Score  
- Sleep Hours  
- Attendance Percentage  
- Stress Level  
- Phone Usage Hours  
- Social Media Hours  

### 🎯 Target

- **Productivity Score (Out of 100)**  

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Streamlit  

---

## 📈 Evaluation Metrics

- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- R² Score  
- Accuracy (approximate)  

---

## 🧮 Weighted Ensemble Formula

Final prediction is calculated using:

Weight = (R² + Accuracy + 1/(1+MAE) + 1/(1+MSE)) / 4

Final Score = Σ(Prediction × Weight) / Σ(Weight)


---

## 💻 Features

- Input student attributes  
- Predict using 4 models  
- Compare model outputs  
- Display evaluation metrics  
- Show final reliable productivity score  
- Identify best performing model  

---

## 🚀 How to Run

### 1. Clone the Repository

git clone https://github.com/your-username/student-productivity-predictor.git

cd student-productivity-predictor


### 2. Install Dependencies


pip install -r requirements.txt


### 3. Run the App

streamlit run app.py



---

## 📌 Workflow

1. Data preprocessing  
2. Feature engineering  
3. Model training  
4. Model evaluation  
5. Model saving  
6. Streamlit deployment  

---

## ⚠️ Notes

- Output is **productivity score out of 100**  
- Models are regression-based  
- Accuracy is derived metric  

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to analyze student behavior and predict productivity. The weighted ensemble approach ensures more reliable predictions by combining multiple models.

---

## 👨‍💻 Author
- Sparsh Agrawal
- Aditya Garg
