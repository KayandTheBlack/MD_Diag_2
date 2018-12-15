import numpy as np  # Llibreria matemÃ tica
import pandas as pd
import sklearn  # Llibreia de DM
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

# Load train data
train = pd.read_csv("Train.csv", index_col=0)
X_train = train.drop("readmitted", axis=1)
y_train = train["readmitted"]

# Split train data into train1 and validation
X_train1, X_validation, y_train1, y_validation = train_test_split(X_train, y_train, random_state=42,test_size=0.3)

# Balance train1 data
# Join X and y into same dataset
Train1_Balanced = X_train1
Train1_Balanced['readmitted'] = y_train1

# Check number of instances of the classes
i_class0 = np.where(Train1_Balanced['readmitted'] == 0)[0]
i_class1 = np.where(Train1_Balanced['readmitted'] == 1)[0]

print("class0 = " + str(len(i_class0)))
print("class1 = " + str(len(i_class1)))
print(Train1_Balanced.shape)

# Balance class 1 to get the same number of cases in both classes
# Use a fraction so it can be replicated later when using the original train and test
Train1_Balanced = Train1_Balanced.drop(Train1_Balanced.query('readmitted == 1').sample(frac=0.87108, random_state=1).index)

i_class0 = np.where(Train1_Balanced['readmitted'] == 0)[0]
i_class1 = np.where(Train1_Balanced['readmitted'] == 1)[0]

print("class0 = " + str(len(i_class0)))
print("class1 = " + str(len(i_class1)))
print(Train1_Balanced.shape)

# Separate into X and y again
X_train1_balanced = Train1_Balanced.drop("readmitted", axis=1)
y_train1_balanced = Train1_Balanced["readmitted"]

# Load test data
test = pd.read_csv("Test.csv", index_col=0)
X_test = test.drop("readmitted", axis=1)
y_test = test["readmitted"]


# Check some shapes
print(X_train1_balanced.shape)
print(y_train1_balanced.shape)
print(X_validation.shape)
print(y_validation.shape)


# Get best ki with balanced train1 data
lr = []
lr1 = []
best_recall = 0
best_ki = 0
print("------- BALANCED -------")
for ki in range(1,30,2):
    knc = nb.KNeighborsClassifier(n_neighbors=ki)
    knc.fit(X_train1_balanced, y_train1_balanced)
    print("accuracy with " + str(ki) + " neighbors: " + str(knc.score(X_validation, y_validation)))
    pred = knc.predict(X_validation)
    print(sklearn.metrics.confusion_matrix(y_validation, pred))
    print(metrics.classification_report(y_validation, pred))
    # Save ki with the best recall score for class 0
    if sklearn.metrics.recall_score(y_validation, pred, average=None)[0] > best_recall:
        best_recall = sklearn.metrics.recall_score(y_validation, pred, average=None)[0]
        best_ki = ki
    lr.append(sklearn.metrics.recall_score(y_validation, pred, average=None)[0])
    lr1.append(sklearn.metrics.recall_score(y_validation, pred, average=None)[1])

# Get plot from both classes' recall
plt.plot(range(1,30,2),lr,'g',label='Class 0')
plt.plot(range(1,30,2),lr1,'r',label='Class 1')
plt.xlabel('k')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()

plt.show()


# Best ki with original train and test data
print("---- BEST RESULTS ----")

# Balance original train data with the same proportion as train1
Train_Balanced = train.drop(train.query('readmitted == 1').sample(frac=0.87108, random_state=1).index)

i_class0 = np.where(Train_Balanced['readmitted'] == 0)[0]
i_class1 = np.where(Train_Balanced['readmitted'] == 1)[0]

print("class0 = " + str(len(i_class0)))
print("class1 = " + str(len(i_class1)))
print(Train_Balanced.shape)

X_train_balanced = Train_Balanced.drop("readmitted", axis=1)
y_train_balanced = Train_Balanced["readmitted"]

knc = nb.KNeighborsClassifier(n_neighbors=best_ki)
knc.fit(X_train_balanced, y_train_balanced)
print("accuracy with " + str(best_ki) + " neighbors: " + str(knc.score(X_test, y_test)))
pred = knc.predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print(metrics.classification_report(y_test, pred))