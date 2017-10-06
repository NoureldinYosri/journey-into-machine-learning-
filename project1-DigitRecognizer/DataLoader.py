import numpy as np;
import tensorflow as tf;
import pandas as pd
import csv
import joblib
from sklearn.model_selection import train_test_split

def LoadData(path='../../data/DigitRecognizer/train.csv'):
    f = pd.read_csv(path,delimiter=',')
    return f;

def getPKLData(path = '../../data/DigitRecognizer/smallTDS.pkl'):
    return joblib.load(path)

def splitData(df,test_size = 0.3):
    df = df.as_matrix();
    X = df[:,1:];
    Y = df[:,0];
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size);
    X_train = np.transpose(X_train);
    X_test = np.transpose(X_test);
    Y_train = np.reshape(Y_train,(1,len(Y_train)));
    Y_test = np.reshape(Y_test,(1,len(Y_test)));
    return X_train,X_test,Y_train,Y_test;