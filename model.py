import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import acquire
import explore

from math import sqrt

# hypothesis testing
from scipy import stats

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans

seed = 21


def drop_infinite(X_train,X_val,X_test,y_train,y_val,y_test):
    
    ind = X_train[X_train.chlorides==0].index
    X_train = X_train.drop(ind)
    y_train = y_train.drop(ind)
    
    ind = X_val[X_val.chlorides==0].index
    X_val = X_val.drop(ind)
    y_val = y_val.drop(ind)
    
    ind = X_test[X_test.chlorides==0].index
    X_test = X_test.drop(ind)
    y_test = y_test.drop(ind)
    
    return X_train,X_val,X_test,y_train,y_val,y_test

# y_train[target], y_train.baseline, X_train, X_val
def run_models(X_train,X_val,X_test,y_train,y_val,target): 
    y_train, y_val = establish_baseline(y_train,y_val)
    
    
    # Create a dict to store rmse values from our models
    mod = ['baseline']
    rmse = [explore.calc_rmse(y_train[target],y_train.baseline)]
    rmse_val = [explore.calc_rmse(y_val[target],y_val.baseline)]

    # Create all of our models
    models= [LinearRegression(),LassoLars(max_iter=1000, alpha=.1),TweedieRegressor(power=0),TweedieRegressor(power=1)]
    mod_name = ['lm','lass','tweedie0','tweedie1']
    add_pf = ['_pf2','_pf3']

    for m,n in zip(models,mod_name): #Creates models off our reg X_train/val sets
        a,b,c = model(m,n,X_train,X_val,y_train,y_val,target)
        mod.append(c) # Append values to our lists
        rmse.append(a)
        rmse_val.append(b)
    
    X_train_pf2,X_val_pf2,X_test_pf2,X_train_pf3,X_val_pf3,X_test_pf3 = transform_poly(X_train,X_val,X_test)
   
    pfx = [X_train_pf2,X_train_pf3] #Creates models off of our pf2/pf3 X sets
    pfy = [X_val_pf2,X_val_pf3]


    for x,y,na in zip(pfx,pfy,add_pf):
        for m,n in zip(models,mod_name):
            name = n+na
            a,b,c = model(m,name,x,y,y_train,y_val,target) # Returns rmse_train, rmse_validate, mod_name
            mod.append(c) # Append mod_name
            rmse.append(a) # Append rmse_train
            rmse_val.append(b) #Append rmse_validate
    
    df = pd.DataFrame({'model':mod,'rmse_train':rmse,'rmse_val':rmse_val}) # Create df out of listsb
    df['difference'] = df['rmse_train'] - df['rmse_val']

    return df

def model_test(ml_model,mod_name,X_test): #Create a function to check rmse on test set
    model = ml_model
    model.fit(X_train_pf3,y_train.quality)
    pred = model.predict(X_test).round(1)
    rmse_test =calc_rmse(y_test.quality,pred)
    
    return rmse_test

def model(ml_model,mod_name,X_train,X_val,y_train,y_val,target): #Create model function calculate rmse on train and validate
    '''
    ml_model: ml model
    mod_name: name of the model
    X_train: X dataset
    X_val: X dataset
    y_train: 
    y_val
    target = dependent variable
    '''
    model = ml_model
    model.fit(X_train,y_train[target])
    pred = model.predict(X_train).round(1)
    rmse_train =explore.calc_rmse(y_train[target],pred)
    pred = model.predict(X_val).round(1)
    rmse_validate = explore.calc_rmse(y_val[target],pred)
    
    return rmse_train,rmse_validate, mod_name

def transform_poly(X_train,X_val,X_test):
    # Transform our X_train and X_val set into polynomials of range 2-3
    pf2 = PolynomialFeatures(degree=2)
    X_train_pf2 = pf2.fit_transform(X_train)
    X_val_pf2 = pf2.fit_transform(X_val)
    X_test_pf2 = pf2.fit_transform(X_test)

    pf3 = PolynomialFeatures(degree=3)
    X_train_pf3 = pf3.fit_transform(X_train)
    X_val_pf3 = pf3.fit_transform(X_val)
    X_test_pf3 = pf3.fit_transform(X_test)
    
    return X_train_pf2,X_val_pf2,X_test_pf2,X_train_pf3,X_val_pf3,X_test_pf3

def establish_baseline(y_train,y_val):
    '''
    Establishes a baseline RMSE and returns y_train and y_val with the baseline RMSE
    '''
    # Establish baseline predictions 
    y_train['baseline'] = round(y_train.quality.mean(),1)
    # Evaluate baseline on validate test
    y_val['baseline'] = round(y_train.quality.mean(),1)
    return y_train, y_val