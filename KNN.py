import numpy as np  # Llibreria matemÃ tica
import matplotlib.pyplot as plt  # Per mostrar plots
import sklearn  # Llibreia de DM
import sklearn.datasets as ds  # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv  # Pel Cross-validation
import sklearn.neighbors as nb  # Per fer servir el knn
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif


# Load the data
train = pd.read_csv("Train2.csv", index_col=0)
test = pd.read_csv("Test2.csv", index_col=0)
X_train = train.drop("readmitted", axis=1).as_matrix()
y_train = train["readmitted"].as_matrix()
X_test = test.drop("readmitted", axis=1).as_matrix()
y_test = test["readmitted"].as_matrix()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

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
scorer = make_scorer(recall_score,average='macro',labels=[0])

params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
knc = nb.KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid=params,cv=10,n_jobs=-1,scoring=scorer)  # If cv is integer, by default is Stratifyed
clf.fit(X_train, y_train)
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