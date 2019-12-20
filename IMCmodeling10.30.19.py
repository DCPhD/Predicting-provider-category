
### TEST DIFFERENT CLASSIFIERS ###

import pandas as pd
from sklearn import neighbors
import numpy as np
from sklearn.model_selection import train_test_split


# load csv files
trainlabelsdf = pd.read_csv("IMCtrainlabels.csv")
trainsetdf = pd.read_csv("IMCtrainset.csv")

# change output settings
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)

#print(trainsetdf.head())

# eliminate EMID column in train set
cols = [col for col in trainsetdf.columns if col not in ["EMID"]]
data = trainsetdf[cols]
target = trainlabelsdf

# split dataset
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=10)

# LinearSVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svc_model = LinearSVC(random_state=0)
pred = svc_model.fit(data_train, target_train).predict(data_test)
print("LinearSVC accuracy: ", accuracy_score(target_test, pred, normalize=True))
# ACCURACY = 0.8656

# K Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_train, target_train)
pred = neigh.predict(data_test)
print("KNeighbors accuracy scores :",accuracy_score(target_test,pred))
# ACCURACY = 0.924

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
pred = gnb.fit(data_train, target_train).predict(data_test)
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize=True))
# ACCURACY = 0.699

### RUN KNN MODEL ###

# read in csv files
df = pd.read_csv("IMCtrainset2.csv")
df.drop(["EMID"], 1, inplace=True)
df2 = pd.read_csv("IMCtestset2.csv")
df2.drop(["EMID"], 1, inplace=True)

# create arrays for x and y axis
X = np.array(df.drop(["OWNER"],1))
y = np.array(df["OWNER"])

# split into test/train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# run KNN algorithm
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# determine accuracy of KNN model
accuracy = clf.score(X_test, y_test)
#print(accuracy)

# run KNN model on test dataset (df2)
example_measures = np.array(df2)
example_measures = example_measures.reshape(len(example_measures), -1)

# predict outcome based on KNN model
prediction = clf.predict(example_measures)
#print(prediction)

# add prediction to existing test dataset
df2["OWNER"] = prediction
#print(df2.head())

print(df2.to_csv("test_dataset_results.csv"))

## COMPARING KNN PREDICTED OWNER TO ACTUAL OWNER = 91.5% ACCURACY ###