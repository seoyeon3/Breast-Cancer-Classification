# Breast Cancer Prediction App

## Problem Definition
Early and accurate classification of breast cancer tumors is critical for effective treatment and improved patient outcomes.

The objective of this project is to build a machine learning model that can classify tumors as malignant or benign based on structured clinical features.

## Live Application 
Users can input clinical features and receive real-time predictions powered by the XGBoost model.
This project is deployed as a Streamlit web application: [![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://breast-cancer-xgboost-app-yu88jyhseihqtfoymsxymm.streamlit.app/)


(*This application is for educational purposes only and not intended for medical use*).


## Project Workflow

1. Defined the classification task using structured medical data  
2. Conducted exploratory data analysis (EDA) to understand feature distributions and relationships  
3. Preprocessed data through feature scaling  
4. Established a baseline model using Logistic Regression  [![Notebook](https://img.shields.io/badge/Notebook-Logistic%20Regression-blue)](https://github.com/seoyeon3/Breast-Cancer-Classification/blob/main/model1_Logistic_Regression.ipynb)
5. Developed an Artificial Neural Network (ANN) to capture non-linear patterns  [![Notebook](https://img.shields.io/badge/Notebook-ANN%20Model-blue)](https://github.com/seoyeon3/Breast-Cancer-Classification/blob/main/model2_ANN.ipynb)
6. Trained and compared XGBoost, selecting it as the final model 
7. Deployed the selected model via a Streamlit application for real-time prediction

## Key Results
- Established a baseline using Logistic Regression (F1: 0.93)
- Improved performance to F1-score of 0.98 using ANN and XGBoost with hyperparameter tuning
- Selected XGBoost for deployment due to comparable performance with ANN and better efficiency and interpretability
- Deployed the final model as an interactive Streamlit application for real-time prediction

## How to Run

```bash
git clone https://github.com/seoyeon3/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
pip install -r requirements.txt
streamlit run app.py
