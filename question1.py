import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy import random
from utils import load_dataset1, load_dataset2
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d

def display(x, y):
    nsamples = x.shape[0]
    fig, axs = plt.subplots(10, 10, figsize=(20,20))
    for i in range(10):
        indices = np.where(y == i)[0]
        count=0
        while count<10:
            index = indices[count]
            image = x[index]
            label = y[index]
            #axs[i,count].title('Label is {label}'.format(label=label))
            axs[i,count].imshow(image, cmap='gray')
            axs[i,count].axis('off')
            count=count+1
    plt.show()

def plotscatterDataset2(x, y):
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = {0:'red',1:'blue',2:'green',3:'yellow'}
    fig, ax = plt.subplots()
    for i in np.unique(y):
        index = np.where(y == i)
        ax.scatter(x1[index], x2[index], c = colormap[i], label = i, s = 100)
    ax.legend()
    plt.show()

def plotscatterDataset1(x, y):
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = {0:'red',1:'orange',2:'blue',3:'green',4:'yellow',5:'purple',6:'pink',7:'grey',8:'darkblue',9:'lightblue'}
    fig, ax = plt.subplots()
    for i in np.unique(y):
        index = np.where(y == i)
        ax.scatter(x1[index], x2[index], c = colormap[i], label = i, s = 100)
    ax.legend()
    plt.show()

def plotscatter3D(x, y):
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    colormap = {0:'red',1:'orange',2:'blue',3:'green',4:'yellow',5:'purple',6:'pink',7:'grey',8:'darkblue',9:'lightblue'}
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in np.unique(y):
        index = np.where(y == i)
        ax.scatter(x1[index], x2[index], x3[index], c = colormap[i], label = i, s = 100)
    ax.legend()
    plt.show()

#part a) Code to display 10 images per class

x, y = load_dataset1()
print(x.shape)
print(y.shape)
display(x,y)

#part b) Code to display scatter plot of dataset_2
x, y = load_dataset2()
print(x.shape)
print(y.shape)
print(np.unique(y))
plotscatterDataset2(x, y)

#part c) Code to display tsne plot 2D of dataset_1
x, y = load_dataset1()
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x = x[:500,:]
y = y[:500]
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)
plotscatterDataset1(tsne_results, y)

#part d) Code to display tsne plot 3D of dataset_1
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)
plotscatter3D(tsne_results, y)