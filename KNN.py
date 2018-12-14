import numpy as np  # Llibreria matemÃ tica
import pandas as pd
import sklearn  # Llibreia de DM
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV
from statsmodels.stats.proportion import proportion_confint

# Load the data


# # Simple cross-validation
# # f-score 0.08 and 0.93

# # Create a kNN classifier object
# knc = nb.KNeighborsClassifier()
#
# # Train the classifier
# knc.fit(X_train, y_train)
#
# # Obtain accuracy score of learned classifier on test data
# print(knc.score(X_test, y_test))
#
# y_pred = knc.predict(X_test)
# print(sklearn.metrics.confusion_matrix(y_test, y_pred))
#
# # Obtain Recall, Precision and F-Measure for each class
# print(metrics.classification_report(y_test, y_pred))

# Grid Search
# COMMENT IF YOU WANT TO CHECK THE BEST RESULT ONLY. TAKES TOO LONG
scorer = make_scorer(f1_score,average='macro',labels=[0]) #TOCAR ESTO!!!!!!!!!!!!!!

params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
knc = nb.KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid=params,cv=[(slice(None), slice(None))],n_jobs=-1,scoring=scorer)  # If cv is integer, by default is Stratifyed
print("DESPRES DE CV")

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
print("ABANS DE FIT")
clf.fit(X_train, y_train)
print("DESPRES DE FIT")
print("Best Params=",clf.best_params_, "Accuracy=", clf.best_score_)

parval=clf.best_params_
knc = nb.KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])
knc.fit(X_train, y_train)
pred=knc.predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred))

# Interval confidence
epsilon = sklearn.metrics.accuracy_score(y_test, pred)
print("Can approximate by Normal Distribution?: ",X_test.shape[0]*epsilon*(1-epsilon)>5)
print("Interval 95% confidence:", "{0:.3f}".format(epsilon), "+/-", "{0:.3f}".format(1.96*np.sqrt(epsilon*(1-epsilon)/X_test.shape[0])))
# or equivalent
print(proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='normal'))



# Trying to remove noise
fs = SelectKBest(mutual_info_classif, k=64).fit(X_train, y_train) #chi2
X_new = fs.transform(X_train)
Xtr_new = fs.transform(X_test)

# COMMENT IF YOU WANT TO CHECK THE BEST RESULT ONLY. TAKES TOO LONG
for ki in range(1,30,2):
    knc = nb.KNeighborsClassifier(n_neighbors=ki)
    knc.fit(X_new, y_train)
    print("accuracy with " + str(ki) + " neighbors: " + str(knc.score(Xtr_new, y_test)))
    pred = knc.predict(Xtr_new)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print(sklearn.metrics.accuracy_score(y_test, pred))
    print(metrics.classification_report(y_test, pred))



# Best result
ki = 21
knc = nb.KNeighborsClassifier(n_neighbors=ki)
knc.fit(X_new, y_train)
print("accuracy with " + str(ki) + " neighbors: " + str(knc.score(Xtr_new, y_test)))
pred = knc.predict(Xtr_new)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred))