'''
This module is to predict Customer Churn

Author: J.C
Data: 10.02.2022

'''


import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from PIL import ImageDraw
from PIL import Image



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # plot churn and not churn customers
    fig = plt.figure(figsize=(20,10))
    df['Churn'].hist()
    fig.savefig('./images/eda/hist_churn.png')
    
    #plot histgram of customers' age
    fig = plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    fig.savefig('./images/eda/hist_age.png')
    
    # barplot of customers' material status
    fig = plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig.savefig('./images/eda/bar_material_status.png')
    
    # distribution of total transaction count 
    fig = plt.figure(figsize=(20,10))
    sns.distplot(df['Total_Trans_Ct'])
    fig.savefig('./images/eda/distplot_Total_Trans_Ct.png')
    
    # create heatmap  
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig('./images/eda/heatmap.png')
    
def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    for cat in category_lst:
        df[str(cat)+'_Churn'] = [df.groupby(cat).mean()['Churn'].loc[val] for val in df[cat]]
    return df

        


def perform_feature_engineering(df,x_cols,y_col):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[x_cols] = df[x_cols]
    y = df[y_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

        
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.savefig('./images/classfication_report/Random_Forest_Train_Test.png')
    print('Save Random Forest report figure')

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature  importances
    feature_names = [X_data.columns[i] for i in indices]
    
    # Create plot
    fig=plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), feature_names, rotation=90)
    fig.savefig(output_pth)

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression( solver='saga',max_iter=1000)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    #train random forest
    cv_rfc.fit(X_train, y_train)
    #train logistic regression
    lrc.fit(X_train, y_train)
    
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/plot_roc_curve.png')
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    
    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig('./images/results/TreeExplainer.png')
    
    feature_importance_plot(cv_rfc, X, './images/results/feature_importance_plot.png')
 
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    

if __name__ == "__main__":
    cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    df = cls.import_data("./data/bank_data.csv")
    perform_eda(df)
    df=encoder_helper(df, cat_columns)
    X_train, X_test, y_train, y_test=perform_feature_engineering(df,keep_cols,y_col='Churn')
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    train_models(X_train, X_test, y_train, y_test)
    
    img = Image.new('RGB', (10, 10))
    lr_img = ImageDraw.Draw(img)
    #lr_img.text((10,10), "Classification Report")
    lr_img.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    lr_img.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10},fontproperties = 'monospace') # approach improved by OP -> monospace!
    lr_img.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    lr_img.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace! 
    img.save('images/classfication_report/Logistic_Regression_Train_Test.png')
    print('classfication_report')