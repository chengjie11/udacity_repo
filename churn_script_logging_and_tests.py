import os
import logging
import churn_library as cls
import pandas as pd
 

os.environ['QT_QPA_PLATFORM']='offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		cls.perform_eda(df)
		assert os.path.isfile("./images/eda/hist_churn.png")
		assert os.path.isfile("./images/eda/hist_age.png")
		assert os.path.isfile("./images/eda/bar_material_status.png")
		assert os.path.isfile("./images/eda/distplot_Total_Trans_Ct.png")
		assert os.path.isfile("./images/eda/heatmap.png")
		logging.info('Test perform_eda: SUCCEED')
	except AssertionError as err:
		logging.info('Test perform_eda: Faild')
		raise err
 
    

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	try:
		assert set(cat_columns).issubset(df.columns)
		logging.info('Test encoder_helper: the original column is contained in the dataframe')
	except AssertionError as err:
		logging.error("Testing encoder_helper: the columns is wrong")
		raise err
        
	try:
		df_encoder = encoder_helper(df, cat_columns)
		logging.info('Test encoder_helper: SUCCESS')
	except AssertionError as err:
		logging.error("Testing encoder_helper: FAILD")
		raise err
        
    
        

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	try:
		X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df,keep_cols,y_col='Churn')
		logging.info('Testing perform_feature_engineering: SUCCESS')
	except Exception as err:
		logging.error("Testing perform_feature_engineering: FAILD")
		raise err

def test_train_models(train_models):
	'''
	test train_models
	'''
	try:
		cls.train_models(X_train, X_test, y_train, y_test)
		assert os.path.isfile('./images/classfication_report/Random_Forest_Train_Test.png')
		#assert os.path.isfile('./images/classfication_report/Logistic_Regression_Train_Test.png')
		assert os.path.isfile('./images/results/plot_roc_curve.png')
		assert os.path.isfile('./images/results/TreeExplainer.png')
		assert os.path.isfile('./images/results/feature_importance_plot.png')
		assert os.path.exists('./models/logistic_model.pkl')
		assert os.path.exists('./models/rfc_model.pkl')
		logging.info('Test train_models: SUCCESS')
	except FileNotFoundError as err:
		logging.error("Testing train model: File not Found")
		raise err    
if __name__ == "__main__":
	cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
	keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
	param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}
	test_import(cls.import_data)
	df = cls.import_data("./data/bank_data.csv")
	test_eda(cls.perform_eda)
	test_encoder_helper(cls.encoder_helper)
    #update df
	df = cls.encoder_helper(df, cat_columns)
	test_perform_feature_engineering(cls.perform_feature_engineering)
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df,keep_cols,y_col='Churn')
	print('X_train',X_train.shape)
	X = pd.DataFrame()
	X[keep_cols] = df[keep_cols]
	print('X[keep_cols]',X.shape)
	test_train_models(cls.train_models)
 


