import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy import random
from utils import load_dataset3, splitDataset3, calculateAccuracy, calculateSklearnAccuracy, generateRandomSplit, preprocessDataset3
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def ensembling(xoriginal, maxdepth, n, xtest, ytest):
    numDecisionTrees = n
    models = []
    train_predictions = []
    test_predictions = []
    for i in range(numDecisionTrees):
        x_random, y_random = generateRandomSplit(xoriginal)
        model = DecisionTreeClassifier(criterion="entropy",max_depth=maxdepth)
        model.fit(x_random, y_random)
        models.append(model)

    for model in models:
        y_train_pred = model.predict(x_random)
        y_test_pred = model.predict(xtest)
        train_predictions.append(y_train_pred)
        test_predictions.append(y_test_pred)

    train_predictionsList = np.array(train_predictions)
    test_predictionsList = np.array(test_predictions)

    y_train_preds = []
    y_test_preds = []
    for i in range(train_predictionsList.shape[1]):
        y_i = train_predictionsList[:,i]
        y_train_preds.append(np.bincount(y_i).argmax())
    for i in range(test_predictionsList.shape[1]):
        y_i = test_predictionsList[:,i]
        y_test_preds.append(np.bincount(y_i).argmax())

    y_trainpreds = np.array(y_train_preds)
    y_testpreds = np.array(y_test_preds)
    trainaccuracy = calculateAccuracy(y_random,y_trainpreds)
    testaccuracy = calculateAccuracy(ytest,y_testpreds)
    return trainaccuracy, testaccuracy


#Loading dataset_3
x = load_dataset3()
#Preprocessing
xoriginal = preprocessDataset3(x)

# print("Nan values year", sum(pd.isnull(x['year'])))
# print("Nan values day", sum(pd.isnull(x['day'])))
# print("Nan values hour", sum(pd.isnull(x['hour'])))
# print("Nan values pm2.5", sum(pd.isnull(x['pm2.5'])))
# print("Nan values dewp", sum(pd.isnull(x['DEWP'])))
# print("Nan values temp", sum(pd.isnull(x['TEMP'])))
# print("Nan values pres", sum(pd.isnull(x['PRES'])))
# print("Nan values cbwd", sum(pd.isnull(x['cbwd'])))
# print("Nan values Iws", sum(pd.isnull(x['Iws'])))
# print("Nan values Is", sum(pd.isnull(x['Is'])))
# print("Nan values Ir", sum(pd.isnull(x['Ir'])))

# #Checking if replacing null values with mean of column would give better accuracy
# mean_ = xoriginal['pm2.5'].mode()
# xoriginal['pm2.5'] = xoriginal['pm2.5'].fillna(mean_)
# print("Nan values", sum(pd.isnull(x['pm2.5'])))

print(xoriginal.head())

y = xoriginal['month']
x = xoriginal.drop(['month'],axis=1)

print("X shape ",x.shape)
print("Y shape ",y.shape)

# uniqueValues = x['cbwd'].unique()
# print(uniqueValues)

x_train, x_test, y_train, y_test = splitDataset3(x, y, 0.8)

print("Nan values", sum(pd.isnull(x_train['pm2.5'])))

dt_entropy = DecisionTreeClassifier(criterion = "entropy") 
#print(dt_entropy)
dt_entropy.fit(x_train, y_train)
y_pred = dt_entropy.predict(x_test)
print("Calculated accuracy on Entropy Decision Tree: ",calculateAccuracy(y_test, y_pred))

dt_gini = DecisionTreeClassifier(criterion = "gini")
#print(dt_gini)
dt_gini.fit(x_train, y_train)
y_pred = dt_gini.predict(x_test)
print("Calculated accuracy on Gini Index Decision Tree: ",calculateAccuracy(y_test, y_pred))

depths = [2,4,8,10,15,30]
# print(depths)
l1 = []
l2 = []
l3 = []
best_acc_ent = 0
optimal_depth_ent = 0
for i in depths:
    depth = i
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = depth)
    # Training 
    clf_entropy.fit(x_train, y_train)

    y_train_pred = clf_entropy.predict(x_train)
    y_test_pred = clf_entropy.predict(x_test)

    train_acc = calculateAccuracy(y_train, y_train_pred)
    test_acc = calculateAccuracy(y_test, y_test_pred)

    if(test_acc>best_acc_ent):
        best_acc_ent=test_acc
        optimal_depth_ent=depth
    l1.append(depth)
    l2.append(test_acc)
    l3.append(train_acc)

print("Depth ",l1)
print("Training Accuracy ",l3)
print("Testing Accuracy ",l2)

print("Best accuracy of entropy based Decision tree {} at optimal depth of {}".format(best_acc_ent,optimal_depth_ent))
plt.plot(l1,l3, label="Training Accuracy")
plt.plot(l1,l2, label="Testing Accuracy")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("Entropy Based Decision Tree Accuracy vs Depth")
plt.legend()
plt.show()

l1 = []
l2 = []
l3 = []
best_acc_ent = 0
optimal_depth_ent = 0
for i in depths:
    depth = i
    clf_entropy = DecisionTreeClassifier(criterion = "gini", max_depth = depth)
    # Training 
    clf_entropy.fit(x_train, y_train)

    y_train_pred = clf_entropy.predict(x_train)
    y_test_pred = clf_entropy.predict(x_test)

    train_acc = calculateAccuracy(y_train, y_train_pred)
    test_acc = calculateAccuracy(y_test, y_test_pred)

    if(test_acc>best_acc_ent):
        best_acc_ent=test_acc
        optimal_depth_ent=depth
    l1.append(depth)
    l2.append(test_acc)
    l3.append(train_acc)

print("Depth ",l1)
print("Training Accuracy ",l3)
print("Testing Accuracy ",l2)

print("Best accuracy of Gini based Decision tree {} at optimal depth of {}".format(best_acc_ent,optimal_depth_ent))
plt.plot(l1,l3, label="Training Accuracy")
plt.plot(l1,l2, label="Testing Accuracy")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("Gini Based Decision Tree Accuracy vs Depth")
plt.legend()
plt.show()

trainaccuracy, testaccuracy = ensembling(xoriginal, 3, 100, x_test, y_test)

print("The accuracy after ensembling technique: ",trainaccuracy, testaccuracy)

#Tuning maxdepth of decision stumps and number of decision trees
depths = [4,8,10,15,20]
best_acc_train = 0
optimal_max_depth_train = 0
optimal_num_train = 0
best_acc_test = 0
optimal_max_depth_test = 0
optimal_num_test = 0
print("Checking optimal configuration ..")
for i in depths:
    for j in range(100):
        trainresult, testresult = ensembling(xoriginal, i, j+1, x_test, y_test)
        if(trainresult > best_acc_train):
            best_acc_train = trainresult
            optimal_max_depth_train = i
            optimal_num_train = j+1
        if(testresult > best_acc_test):
            best_acc_test = testresult
            optimal_max_depth_test = i
            optimal_num_test = j+1
    print("The best training accuracy obtained at maxdepth {} with {} stumps is {}".format(i,optimal_num_train,best_acc_train))
    print("The best testing accuracy obtained at maxdepth {} with {} stumps is {}".format(i,optimal_num_test,best_acc_test))
print("After tuning, the best training accuracy obtained is {} at optimal max depth {} using {} decision stumps".format(best_acc_train,optimal_max_depth_train,optimal_num_train))
print("After tuning, the best testing accuracy obtained is {} at optimal max depth {} using {} decision stumps".format(best_acc_test,optimal_max_depth_test,optimal_num_test))
