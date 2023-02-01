import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

seed = 21

def outlier_detector(df,column,k=1.5): #Run assign_outlier, not this one
    q1,q3 = df[column].quantile([.25,.75])
    iqr = q3 - q1
    upper_bound = q3 +k*iqr
    lower_bound = q1 -k*iqr
    print(column, lower_bound,upper_bound)
    return np.where(df[column]> upper_bound,1,0), np.where(df[column]< lower_bound,1,0)

def assign_outlier(df, cols):
    '''
    Returns a new df with outlier detection for passed in columns
    '''
    for col in cols:
        df[f'{col}_upper_outliers'],df[f'{col}_lower_outliers'] = outlier_detector(df,col)
    return df

def explore_nulls(df):
    null_count = df.isnull().sum()
    null_perc = df.isnull().sum()/df.shape[0]
    return pd.DataFrame({'null_count':null_count,'null_perc':null_perc})

def handle_missing_values(df, prop_required_column, prop_required_row):
    temp = explore_nulls(df) 
    cols_drop = temp[temp.null_percent>prop_required_column]['column_name'].values.tolist()
    df.drop(cols_drop,axis=1)

    df['row_missing'] = df.isnull().sum(axis=1)/df.shape[1] #These two lines take care of the rows
    df.drop(df[df.row_missing>.4].index)
    del df.row_missing
        
    return df

def print_outliers(df):
    outliers_cols = [col for col in df.columns if 'outliers' in col]
    return df[df[outliers_cols].any(axis=1)==1]

def split_data(df, target):
    '''
    Splits a df into a train, validate, and test set. 
    target is the feature you will predict
    '''
    train_validate, test = train_test_split(df, train_size =.8, random_state = 21)
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 21)
    X_train = train.drop(columns=target)
    y_train = train[target]
    
    X_val = validate.drop(columns=target)
    y_val = validate[target]
    X_test = test.drop(columns=target)
    y_test = test[target]
    
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return train, X_train, y_train, X_val, y_val, X_test, y_test

def scale_minmax(X_train,X_val,X_test):
    '''
    Takes in train, validate, and test sets and returns the minmax scaled dfs
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
    X_val[X_val.columns] = scaler.transform(X_val[X_val.columns])
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    
    return X_train,X_val,X_test

def plot_inertia(df):
    inertia = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    
    clust_inert = pd.DataFrame({'n_clusters':list(range(1,10)),'inertia':inertia})
    
    return sns.relplot(data=clust_inert,x='n_clusters',y='inertia',kind='line')

def plot_kmeans_cluster(df,n):  
    kmeans = KMeans(n_clusters=n, random_state=seed)
    kmeans.fit(df)

    sns.relplot(data=df,x=df.iloc[:,0],y=df.iloc[:,1],hue=pd.Series(kmeans.predict(df)))