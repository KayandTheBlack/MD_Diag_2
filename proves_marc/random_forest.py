from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def filterp(th,ProbClass1):
    """ Given a treshold "th" and a set of probabilies of belonging to class 1 "ProbClass1", return predictions """
    y=np.zeros(ProbClass1.shape[0])
    for i,v in enumerate(ProbClass1):
        if ProbClass1[i]>th:
            y[i]=1
    return y


train = pd.read_csv("TRAIN.csv",index_col=0)
test = pd.read_csv("TEST.csv",index_col=0)


X = train.drop("TARGET", axis=1)
Y = train["TARGET"]
X_test = test.drop("TARGET", axis=1)
Y_test = test["TARGET"]

clf = RandomForestClassifier(n_estimators = 100)#, max_depth=25, criterion = "gini", min_samples_split=10)
clf.fit(X, Y)
pred=clf.predict(X_test)
print(classification_report(Y_test, pred))
print("accuracy = " + str(accuracy_score(Y_test, pred)))
clf = RandomForestClassifier(n_estimators = 100)#, max_depth=25, criterion = "gini", min_samples_split=10)
clf.fit(X, Y)
probs=clf.predict_proba(X_test)
print(probs)
for th in range(4):
    print('\n', th*0.1)
    npred = filterp(th*0.1,probs[:,0])
    print(classification_report(Y_test, npred))
    print(confusion_matrix(Y_test, npred))