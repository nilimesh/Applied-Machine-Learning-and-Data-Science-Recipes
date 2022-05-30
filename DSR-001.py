# ------------------------------
# SETScholars: DSR-001.py ------
# ------------------------------

# -*- coding: utf-8 -*-
############################################################################
# This is an end-2-end Applied Machine Learning Script with RDBMS
# Title: IRIS Flower Classification using Logistic Regression Classifier
# Knowledge required: Basic Python, Scikit-Learn and MySQL
# System requirements:
#   a) Python (3.X) distribution from Anaconda (Anaconda 3)
#   b) MySQL 5.7 with an user: root and password:
############################################################################

"""
@author: 
    Nilimesh Halder, PhD
    BSc in Computer Science and Engineering, 
        @ Khulna University, Bangladesh.
    PhD in Artificial Intelligence and Applied Machine Learning, 
        @ The University of Western Australia, Australia.
"""

# -----------------------------------------------------------------------------
# Steps in Applied Machine Learning:
# 1. Load Library
# 2. Load Dataset to which Machine Learning Algorithm to be applied
#    Either a) load from a CSV file or b) load from a Database   
# 3. Summarisation of Data to understand dataset (Descriptive Statistics)
# 4. Visualisation of Data to understand dataset (Plots, Graphs etc.)
# 5. Data pre-processing & Data transformation (split into train-test datasets)
# 6. Application of a Machine Learning Algorithm to training dataset 
#   a) setup a ML algorithm and parameter settings
#   b) cross validation setup with training dataset
#   c) training & fitting Algorithm with training Dataset
#   d) evaluation of trained Algorithm (or Model) and result
#   e) saving the trained model for future prediction
# 7.             
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# 1. Load necessary libraries
import sqlalchemy as sa
import pandas as pd
import pickle as pk
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

##Load DataSet from CSV file
def loadFrCSVFile(filename):
    print(filename)
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    dataset = pd.read_csv(filename, names=col_names)
    return dataset
    
##Import DataSet to a MySQL Database
def import2MySQL(dataset):
    engine_str = (
            'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
            user      =  'root',
            password  =  'root888',
            server    =  'localhost',
            database  =  'DataScienceRecipes'))
    
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else:      print("MySQL Connection is not Successful ... ... ...")
    
    dataset.to_sql(name='irisdata', con=engine, schema='datasciencerecipes', 
                   if_exists = 'replace', chunksize = 1000, index=False)
    conn.close()
    
##Load DataSet from MySQL Database to Pandas a DataFrame
def loadDataSetFrMySQLTable():
    engine_str = (
            'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
            user      =  'root',
            password  =  'root888',
            server    =  'localhost',
            database  =  'datasciencerecipes'))
    
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else:      print("MySQL Connection is not Successful ... ... ...")
    
    # MySQL Query with few generated Attributes/Features
    query = '''
    SELECT  sepal_length, 
            sepal_width, 
            petal_length, 
            petal_width, 
            round(sepal_length/sepal_width,2) as ratio1, 
            round(sepal_width/petal_length,2) as ratio2,
            round(petal_length/petal_width,2) as ratio3,
            round(petal_width/sepal_length,2) as ratio4,
            round(sepal_width/sepal_length,2) as ratio5, 
            round(petal_length/sepal_width,2) as ratio6,
            round(petal_width/petal_length,2) as ratio7,
            round(sepal_length/petal_width,2) as ratio8,
            class 
    FROM irisdata;
    '''
    query_result = conn.execute(query)
    dataset =  pd.DataFrame(query_result.fetchall(), 
                            columns =  query_result.keys())
    print('DataFrame Size',dataset.shape);
    print('ROW',dataset.shape[0]);print('COLUMN',dataset.shape[1]);
    conn.close()
    return dataset

## Data Summarisation (Descriptive Statistics)
def summariseDataset(dataset):
    cols1 = ['sepal_length','sepal_width','petal_length','petal_width']
    cols2 = ['ratio1','ratio2','ratio3','ratio4']
    cols3 = ['ratio5','ratio6','ratio7','ratio8']    
    # shape
    print(dataset[cols1].shape)
    print(dataset[cols2].shape)
    print(dataset[cols3].shape)    
    # head
    print(dataset[cols1].head(5))
    print(dataset[cols2].head(5))
    print(dataset[cols3].head(5))    
    # descriptions
    print(dataset[cols1].describe())
    print(dataset[cols2].describe())    
    print(dataset[cols3].describe())
    # class distribution
    print(dataset.groupby('class').size())

## Data Visualisation to understand Data
def visualiseDataset(dataset):
    cols1 = ['sepal_length','sepal_width','petal_length','petal_width']
    cols2 = ['ratio1','ratio2','ratio3','ratio4']
    cols3 = ['ratio5','ratio6','ratio7','ratio8'] 
    
    # box and whisker plots
    dataset[cols1].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()
    dataset[cols2].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()
    dataset[cols3].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()    
    # histograms
    dataset[cols1].hist()
    pyplot.show()
    dataset[cols2].hist()
    pyplot.show()
    dataset[cols3].hist()
    pyplot.show()    
    # scatter plot matrix
    scatter_matrix(dataset[cols1])
    pyplot.show()
    scatter_matrix(dataset[cols2])
    pyplot.show()
    scatter_matrix(dataset[cols3])
    pyplot.show()    

## Data Pre-Processing
def preProcessingData(dataset):

    # 1. Data Cleaning
      # There is no missing value. 
      # We could "Outlier treatment" but nothing was done here.  

    # 2. Feature Selection
    cols_X = ['sepal_length','sepal_width','petal_length','petal_width',
              'ratio1','ratio2','ratio3','ratio4',
              'ratio5','ratio6','ratio7','ratio8']
    cols_Y = 'class'
    seed = 7

    # 3. Data Transform - Split out train : test datasets
    train_X, test_X, train_Y, test_Y = train_test_split(dataset.loc[:, cols_X], 
                                                        dataset.loc[:, cols_Y], 
                                                        test_size=0.20,
                                                        random_state = seed
                                                        )
    return train_X, test_X, train_Y, test_Y

## Applied Machine Learning Algorithm ... ... ...
def evaluateAlgorithm(train_X, test_X, train_Y, test_Y):
    
    ##Machine Lreaning Algorithm, Parameter setting 
    model_LR = LogisticRegression()
    
    ##Cross Validation
    cv_results = cross_val_score(model_LR, train_X, train_Y, cv = 4, 
                                 scoring='accuracy', n_jobs = -1, verbose = 0)
    
    print("\nScores from cross validation: ", cv_results)
    print("Mean accuracy score from Cross Validation", cv_results.mean())
    print("Std from Cross Validation", cv_results.std())

    ##Training & Fitting Algorithm with training Dataset
    trained_Model = model_LR.fit(train_X, train_Y)

    ##Evaluation of trained Algorithm (or Model) and result
    pred_Class          = trained_Model.predict(test_X)
    acc         = accuracy_score(test_Y, pred_Class)
    classReport = classification_report(test_Y, pred_Class)
    confMatrix  = confusion_matrix(test_Y, pred_Class) 
    print('\nThe accuracy: {}'.format(acc))
    print('The Classification Report:\n {}'.format(classReport))
    print('The Confusion Matrix:\n {}'.format(confMatrix))
    
    #Save the trained Model
    with open('trainedModel_LR.pickle', 'wb') as f:
        pk.dump(trained_Model, f)

## Load a (new or existing ) dataset to make prediction 
def loadPredictionDataset():
    engine_str = (
            'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
            user      =  'root',
            password  =  'root888',
            server    =  'localhost',
            database  =  'datasciencerecipes'))
    
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else:      print("MySQL Connection is not Successful ... ... ...")

    # MySQL Query - New Query is required for Prediction DataSet
    query = '''
    SELECT  sepal_length, 
            sepal_width, 
            petal_length, 
            petal_width, 
            round(sepal_length/sepal_width,2) as ratio1, 
            round(sepal_width/petal_length,2) as ratio2,
            round(petal_length/petal_width,2) as ratio3,
            round(petal_width/sepal_length,2) as ratio4,
            round(sepal_width/sepal_length,2) as ratio5, 
            round(petal_length/sepal_width,2) as ratio6,
            round(petal_width/petal_length,2) as ratio7,
            round(sepal_length/petal_width,2) as ratio8
    FROM irisdata;
    '''
    query_result = conn.execute(query)
    dataset =  pd.DataFrame(query_result.fetchall(), 
                            columns =  query_result.keys())
    conn.close()
    return dataset

## Load the trained model and make prediction
def loadTrainedModelForPrediction(pred_dataset):
    f = open('trainedModel_LR.pickle', 'rb')
    model = pk.load(f); f.close();
    pred_Class = model.predict(pred_dataset)
    pred_dataset.loc[:, 'classResult'] = pred_Class
    return pred_dataset

## Finalise the results and update the audiance
def finaliseResult(result):

    #Save Result in a CSV file
    print("Save Result in a CSV file ... ... ...")    
    result.to_csv('finalResult.csv', index = False)
    
    #Save Result in a MySQl Table
    engine_str = (
            'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
            user      =  'root',
            password  =  'root888',
            server    =  'localhost',
            database  =  'datasciencerecipes'))
    
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else:      print("MySQL Connection is not Successful ... ... ...")

    print("Save Result in a MySQl Table ... ... ...")    
    result.to_sql(name='irisresult', con=engine, schema='datasciencerecipes', 
                   if_exists = 'replace', chunksize = 1000, index=False)
    conn.close()

# End-to-End Applied Machine Learning Recipes for Beginners and App-Developers
if __name__ == '__main__':
    filename = 'iris.data.csv'
    
    # 2. Load Dataset to which Machine Learning Algorithm to be applied
    dataset = loadFrCSVFile(filename)
    import2MySQL(dataset)
    dataset = loadDataSetFrMySQLTable()
    
    # 3. Summarisation of Data to understand dataset (Descriptive Statistics)
    summariseDataset(dataset)
    
    # 4. Visualisation of Data to understand dataset (Plots, Graphs etc.)
    visualiseDataset(dataset)
    
    # 5. Data pre-processing and Data transformation (split into train-test datasets)
    train_X, test_X, train_Y, test_Y = preProcessingData(dataset)
    
    # 6. Application of a Machine Learning Algorithm to training dataset 
    evaluateAlgorithm(train_X, test_X, train_Y, test_Y)
    
    # 7. Load the saved model and apply it to new dataset for prediction 
    pred_Dataset = loadPredictionDataset()
    result = loadTrainedModelForPrediction(pred_Dataset)
    finaliseResult(result)
    
    print('\nEnd-to-End Applied Machine Learning Recipes for Developers\n')
