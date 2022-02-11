# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project employs a dataset of bank customers with their demographic information and their activity history with the bank account. The goal of this project is to build two classification models - random forest and logistic regression - to predict whether a customer is likely to churn or not. 
Testing and logging is included in building machine learning model in production. Separate test file churn_script_logging_and_tests.py is created and logging function is embedded for debugging purpose.  

## Running Files
Step 1: pip install needed python packages, including shap, joblib, pandas, numpy, matplotlib, seaborn, sklearn
        pip install numpy --upgrade
        pip install -U seaborn when you got issue 'numpy.float64' object cannot be interpreted as an integer

Step 2: run ipython churn_script_logging_and_tests.py and test all function, check open churn_library.log file to check whether there is error shown

Step 3: run ipython churn_library.py to build models and obtain the model outputs

Step 4 (optional): 
       check image folders:
       eda folder includes all plots from exploratory analysis. 
       classfication_report folder includes plots about classification result of random forest and logistic regression. 
       results folder contains plots such as classification reports and feature importance plots.


