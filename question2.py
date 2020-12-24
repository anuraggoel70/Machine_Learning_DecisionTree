import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy import random
from utils import load_dataset2, splitDataset, calculateAccuracy, calculateSklearnAccuracy
from sklearn.tree import DecisionTreeClassifier

x, y = load_dataset2()
print("X shape: ",x.shape)
print("Y shape: ",y.shape)

x_train, x_test, y_train, y_test = splitDataset(x, y, 0.7)

#check occurences of each class in training and testing set
# n_0 = np.count_nonzero(y_test == 0)
# n_1 = np.count_nonzero(y_test == 1)
# n_2 = np.count_nonzero(y_test == 2)
# n_3 = np.count_nonzero(y_test == 3)
# print("Count",n_0,n_1,n_2,n_3)

depths = np.arange(start=1, stop=16, step=1)
depths = np.append(depths,[20,25,30])
#print(type(depths))
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
best_acc = 0
optimal_depth = 0
for i in depths:
    depth = i
    # clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = depth, min_samples_leaf = 5)
    clf_entropy = DecisionTreeClassifier(max_depth = depth)
    #Training 
    clf_entropy.fit(x_train, y_train)

    y_train_pred = clf_entropy.predict(x_train)

    #Calculate training accuracy from implemented accuracy function
    y_train_acc1 = calculateAccuracy(y_train, y_train_pred)
    l1.append(y_train_acc1)

    #Calculate training accuracy from implemented accuracy function
    y_train_acc2 = calculateSklearnAccuracy(y_train, y_train_pred)
    l2.append(y_train_acc2)

    y_test_pred = clf_entropy.predict(x_test)

    #Calculate training accuracy from implemented accuracy function
    y_test_acc1 = calculateAccuracy(y_test, y_test_pred)
    l3.append(y_test_acc1)

    #Calculate training accuracy from implemented accuracy function
    y_test_acc2 = calculateSklearnAccuracy(y_test, y_test_pred)
    l4.append(y_test_acc2)

    #Finding optimal value of depth
    if(y_test_acc1>best_acc):
        best_acc=y_test_acc1
        optimal_depth=depth
    l5.append(depth)

print("Depth ",l5)
print("Training Accuracy using implemented accuracy", l1)
print("Training Accuracy using sklearn accuracy", l2)
print("Testing Accuracy using implemented accuracy", l3)
print("Testing Accuracy using sklearn accuracy", l4)
print("Best testing accuracy {} at optimal depth of {}".format(best_acc,optimal_depth))

fig = plt.figure()
plt.plot(l5,l1)
fig.suptitle('Training Accuracy (Calculated) vs Depth', fontsize=20)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('Training Accuracy (Calculated)', fontsize=16)
# fig.savefig('training1.jpg')
plt.show()

fig = plt.figure()
plt.plot(l5,l2)
fig.suptitle('Training Accuracy (sklearn) vs Depth', fontsize=20)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('Training Accuracy (sklearn)', fontsize=16)
# fig.savefig('training2.jpg')
plt.show()

fig = plt.figure()
plt.plot(l5,l3)
fig.suptitle('Testing Accuracy (Calculated) vs Depth', fontsize=20)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('Testing Accuracy (Calculated)', fontsize=16)
# fig.savefig('testing1.jpg')
plt.show()

fig = plt.figure()
plt.plot(l5,l4)
fig.suptitle('Testing Accuracy (sklearn) vs Depth', fontsize=20)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('Testing Accuracy (sklearn)', fontsize=16)
# fig.savefig('testing2.jpg')
plt.show()