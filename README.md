# 🛍️ Shoplytics Nexus Dashboard

An interactive AI dashboard that analyzes customer shopping behavior and predicts sales amounts using machine learning and Streamlit. Built using Python, scikit-learn, and Plotly.

## 🔍 Project Overview

Shoplytics Nexus combines regression modeling with intuitive data visualization to reveal insights from e-commerce transaction data. The dashboard allows users to:

- Predict sales amounts based on customer and product features
- Filter data by gender and view targeted analytics
- Visualize residuals vs predicted values
- Explore top feature importances using dynamic bar charts
- View distribution of purchase amounts

## 🧠 Technologies Used

- Python 3.13
- scikit-learn
- pandas, numpy
- Streamlit
- Plotly
- joblib (model serialization)

## 📊 Files Included

| File                        | Description                                  |
|----------------------------|----------------------------------------------|
| `app.py`                   | Streamlit dashboard app                      |
| `Shoplytics_Regression.ipynb` | Jupyter Notebook with preprocessing & ML model |
| `Shoplytics_Nexus_Data.csv` | Cleaned e-commerce dataset                  |
| `linear_sales_model.joblib` | Serialized regression model                 |
| `requirements.txt`         | Package dependencies                        |

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/NoorAisha25/Shoplytics_Nexus.git
   cd Shoplytics_Nexus
   pip install -r requirements.txt
   streamlit run app.py
   Then open http://localhost:8501 in your browser!
   
   
