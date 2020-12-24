import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import accuracy_score
import random

def load_dataset1(data_path='dataset_1.mat'):
    data = loadmat(data_path)
    # print(data.keys())
    x = data['samples']
    y = data['labels']

    y = np.transpose(y)
    y = y.reshape(y.shape[0],)

    # print(x.shape) #50000*28*28
    # print(y.shape) #50000
    # print(np.unique(y)) #0..9
    return x, y

def load_dataset2(data_path='dataset_2.mat'):
    data = loadmat(data_path)
    # print(data.keys())
    x = data['samples']
    y = data['labels']

    y = np.transpose(y)
    y = y.reshape(y.shape[0],)

    # print(x.shape) #20000*2
    # print(y.shape) #20000
    # print(np.unique(y)) #0..3
    return x, y

def load_dataset3():
    x = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv', skiprows=0)
    # #Checking if dropping rows having null values would give better accuracy
    #x.dropna(inplace=True)
    return x

def preprocessDataset3(x):
    #Removing No. column"
    x = x.drop(['No'],axis=1)
    #Handling Missing values by replacing null values with 0
    x = x.fillna(0)
    #Converting character values to numeric values
    x['cbwd'] = x['cbwd'].map({'NW':0, 'cv':1, 'NE':2, 'SE':3})
    return x

def splitDataset(x, y, splitratio):
    split = int(splitratio*x.shape[0])

    x_train = x[:split,:]
    x_test = x[split:,:]
    y_train = y[:split]
    y_test = y[split:]
    print("X train shape: ",x_train.shape)
    print("X test shape: ",x_test.shape)
    print("Y train shape: ",y_train.shape)
    print("Y test shape: ",y_test.shape)
    return x_train, x_test, y_train, y_test

def splitDataset3(x, y, splitratio):
    split = int(splitratio*x.shape[0])

    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]
    print("X train shape: ",x_train.shape)
    print("X test shape: ",x_test.shape)
    print("Y train shape: ",y_train.shape)
    print("Y test shape: ",y_test.shape)
    return x_train, x_test, y_train, y_test

def generateRandomSplit(x):
    split = int(0.8*x.shape[0])
    x_train = x[:split]
    # from sklearn.utils import shuffle
    # x_train = shuffle(x_train)
    x_random = x_train.sample(frac=0.5)
    # split = int(0.5*x_train.shape[0])
    # x_random = x[:split]

    y_random = x_random['month']
    x_random = x_random.drop(['month'],axis=1)
    return x_random, y_random

def calculateAccuracy(y_true, y_pred):
    result = np.mean(y_true==y_pred)*100
    result = round(result, 2)
    return result

def calculateSklearnAccuracy(y_true, y_pred):
    result = accuracy_score(y_true,y_pred)*100
    result = round(result, 2)
    return result