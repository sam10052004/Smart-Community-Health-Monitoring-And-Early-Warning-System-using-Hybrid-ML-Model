# Smart Community Health Monitoring and Early Warning System using Hybrid ML Model

## 📌 Overview
This project is a Water Quality Intelligence System that predicts water potability and provides early health risk warnings using a hybrid machine learning approach.

It combines:
- Random Forest
- XGBoost
- LightGBM
- Weighted Ensemble Model

The system integrates machine learning with domain knowledge such as Water Quality Index (WQI) and disease risk prediction to provide meaningful and actionable insights.

---

## 🎯 Objectives
- Predict whether water is potable or not
- Calculate Water Quality Index (WQI)
- Identify contamination levels
- Predict disease risks based on water parameters

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Joblib

---

## 📊 Features Used
- pH  
- Temperature  
- Dissolved Oxygen  
- Conductivity  
- BOD  
- Nitrate  
- Fecal Coliform  
- Total Coliform  
- Hardness  
- Solids  
- Chloramines  
- Sulfate  
- Organic Carbon  
- Trihalomethanes  
- Turbidity  

---

## 🧠 Models Used
- Random Forest (Bagging-based ensemble)
- XGBoost (Gradient Boosting with regularization)
- LightGBM (Efficient histogram-based boosting)
- Weighted Ensemble Model

---

## 📈 Outputs

### 1. Potability Prediction
- Potable (1)
- Not Potable (0)

### 2. Water Quality Index (WQI)
- Single score representing overall water quality

### 3. Contamination Level
- Safe (WQI ≤ 50)
- Moderate (50 < WQI ≤ 100)
- Unsafe (WQI > 100)

### 4. Disease Risk Prediction
- Blue Baby Syndrome (based on nitrate levels)
- Fluorosis (based on mineral content)
- Bacterial Contamination (based on coliform and DO levels)

---

## 🧪 Model Performance

| Model          | Accuracy |
|----------------|----------|
| Random Forest  | ~0.92    |
| XGBoost        | ~0.93    |
| LightGBM       | ~0.93    |
| Ensemble       | ~0.93+   |

---

## 📂 Project Structure

project/
│── src/
│   ├── train.py
│   ├── predict.py
│   ├── input.py
│
│── models/
│   ├── metadata.json
│   ├── features.pkl
│
│── README.md
│── requirements.txt
|

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python src/train.py

### 3. Run prediction
python src/predict.py

---

## 🔬 Key Highlights
- Combines Machine Learning with domain-based calculations
- Uses WQI for real-world interpretation
- Provides health-based insights along with predictions
- Ensemble improves accuracy and reliability

---

## ⚠️ Limitations
- Performance depends on data quality
- No real-time data integration
- Limited hyperparameter tuning
- Prototype-level implementation

---

## 🔮 Future Work
- Real-time IoT integration
- Web or mobile dashboard
- Advanced model optimization
- Larger datasets

---

## 👨‍💻 Author
Your Name

---

## ⭐ Contribution
Feel free to fork this repository and improve the system.

---

## 📌 Repository Description
Hybrid ML system for water quality prediction, WQI calculation, contamination analysis, and disease risk assessment using Random Forest, XGBoost, and LightGBM.
