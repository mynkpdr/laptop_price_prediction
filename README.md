# 💻 Laptop Price Prediction

This repository contains a **machine learning project** to predict laptop prices based on their specifications, along with a deployed **Streamlit application** for interactive usage.

## 1. Project Overview

With the increasing variety of laptop configurations in the market, predicting the price of a laptop based on its specifications is valuable for buyers, sellers, and market analysts. This project:

- Performs **Exploratory Data Analysis (EDA)** on a comprehensive laptop dataset.
- Builds and evaluates a **Random Forest Regressor** model to predict laptop prices.
- Applies **log transformation** to handle right-skewed price distribution.
- Uses a **feature engineering pipeline** with scaling and one-hot encoding for categorical variables.
- Deploys the trained model via **Streamlit** for real-time price prediction.

---

## 2. Repository Structure

```
├── laptop_price_prediction.ipynb   # Notebook/script containing full data analysis, model training, and pipeline creation
├── df_laptop.pkl                # Processed dataset saved for use in the app
├── pipe_laptop.pkl              # Saved Random Forest model pipeline
├── requirements.txt             # Required packages
└── main.py                      # Streamlit app for interactive deployment
```

---

## 3. Features

✅ Exploratory Data Analysis (EDA)  
✅ Data cleaning & preprocessing  
✅ Log transformation of price for normal distribution  
✅ Random Forest Regression model with hyperparameter tuning  
✅ Model performance metrics: RMSE and R²  
✅ Streamlit app for easy user interaction

---

## 4. Model Performance

- **Model:** Random Forest Regressor  
- **R² Score:** ~ 0.8655
- **RMSE:** ~ 0.22

The model performs well with a balanced trade-off between bias and variance, achieving high accuracy on unseen data.

---

## 5. How to Use

1. Clone this repository:

```bash
git clone https://github.com/mynkpdr/laptop-price-prediction.git
cd laptop-price-prediction
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run main.py
```

4. Enter laptop specifications in the Streamlit interface to get the **predicted price instantly**.

---

## 6. Tech Stack

- **Python**  
- **Pandas & NumPy** for data analysis  
- **Scikit-learn** for model building and preprocessing pipeline  
- **Seaborn & Matplotlib** for EDA visualizations  
- **Joblib** for model serialization  
- **Streamlit** for deployment

---

## 7. Acknowledgements

This project was created as part of my **ML portfolio development**, with a focus on end-to-end deployment and practical business use cases.

---
