import numpy as np  # Llibreria matemÃ tica
import pandas as pd
import sklearn  # Llibreia de DM
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt


train = pd.read_csv("Train2.csv", index_col=0)
i_class0 = np.where(train['readmitted'] == 0)[0]
i_class1 = np.where(train['readmitted'] == 1)[0]

print("class0 = " + str(len(i_class0)))
print("class1 = " + str(len(i_class1)))
print(train.shape)
train = train.drop(train.query('readmitted == 0').sample(n=4755).index)
train = train.drop(train.query('readmitted == 1').sample(n=57750).index)
i_class0 = np.where(train['readmitted'] == 0)[0]
i_class1 = np.where(train['readmitted'] == 1)[0]

print("class0 = " + str(len(i_class0)))
print("class1 = " + str(len(i_class1)))
print(train.shape)

test = pd.read_csv("Test2.csv", index_col=0)
X_train = train.drop("readmitted", axis=1).as_matrix()
y_train = train["readmitted"].as_matrix()
X_test = test.drop("readmitted", axis=1).as_matrix()
y_test = test["readmitted"].as_matrix()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Trying to remove noise
fs = SelectKBest(mutual_info_classif, k=64).fit(X_train, y_train) #chi2
X_new = fs.transform(X_train)
Xtr_new = fs.transform(X_test)


lr = []
lr_s = []
for ki in range(1,30,2):
    knc = nb.KNeighborsClassifier(n_neighbors=ki)
    knc_s = nb.KNeighborsClassifier(n_neighbors=ki)

    knc.fit(X_train1, y_train1)
    knc_s.fit(X_new, y_train)

    print("------- ORIGINAL -------")

    print("accuracy with " + str(ki) + " neighbors: " + str(knc.score(X_test, y_test)))
    pred = knc.predict(X_test1)
    print(sklearn.metrics.confusion_matrix(y_test1, pred))
    print(sklearn.metrics.accuracy_score(y_test1, pred))
    print(metrics.classification_report(y_test1, pred))

    print("------- NOISE REMOVED -------")

    print("accuracy with " + str(ki) + " neighbors: " + str(knc_s.score(Xtr_new, y_test)))
    pred_s = knc_s.predict(Xtr_new)
    print(sklearn.metrics.confusion_matrix(y_test, pred_s))
    print(sklearn.metrics.accuracy_score(y_test, pred_s))
    print(metrics.classification_report(y_test, pred_s))

    lr.append(knc.score(X_test, y_test))
    lr_s.append(knc_s.score(Xtr_new, y_test))

plt.plot(range(1,30,2),lr,'r',label='Original')
plt.plot(range(1,30,2),lr_s,'g',label='Noise removed')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()

plt.show()

#tornar a fer classificador amb train normal i millors parametres trobats per recall
# i aplicar sobre test normal
