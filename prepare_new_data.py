import os,sys
import urllib.request as urllib2
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import numpy as np
from random import seed, shuffle
import pandas as pd 
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def check_data_file(fname):
    files = os.listdir(".") # get the current directory listing
    print ("Looking for file '%s' in the current directory..." % fname)

    if fname in files:
        print ("File found in current directory..")
        df = pd.read_csv(fname)
        print('With column names: ', df.head(1))
        print(df.columns)
    
    return

check_data_file("USCensus1990_clean_iClass.csv")    

def load_new_data(load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """
    # run for sex and citizenship respectively 
    attrs = ['dAge', 'iCitizen', 'iDisabl1', 'iDisabl2', 'iEnglish', 'iImmigr', 'iLang1', 'iMarital', 'iRlabor', 'iSchool', 'iSex', 'iYearsch', 'iClass_B'] # all attributes
    sensitive_attrs = ['iSex'] # the fairness constraints will be used for this feature 
    attrs_to_ignore = ['iSex', 'iCitizen','iClass_B'] # sex and citizen aer sensitive features, so no use them in classification 
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # read data 
    df = pd.read_csv("USCensus1990_clean_iClass.csv")[attrs]

    # onehot categorical data
    # print('Before onehot', df.columns)
    for a in attrs_for_classification:
        dummies = pd.get_dummies(df[a], prefix = a)
        df = pd.concat([df, dummies], axis=1).drop(a, axis=1)
    # print('After onehot', df.columns)
    
    # prepare X features, y labels, and x_control sensitive features
    y = df['iClass_B'].tolist() #class binary label
    x_control = df[['iSex', 'iCitizen']].to_dict('list') #sensitive features 
    
    X = df.drop(columns=attrs_to_ignore)
    X = X.to_numpy() 
    print('the number of sample',len(X),'the number of feature attributes', len(X[0]))       

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float)
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control

load_new_data(load_data_size=None)