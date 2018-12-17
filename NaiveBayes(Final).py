import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB  ### Because continuous data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

train = pd.read_csv("Train.csv", index_col=0)
test = pd.read_csv("Test.csv", index_col=0)

df = pd.concat([train,test])
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

print(X.shape)
print(y.shape)



# Manera 1
print("--- Manera 1 ---")
cv = StratifiedKFold(n_splits=10, random_state=1)

gnb = GaussianNB()
cv_scores = cross_val_score(gnb,X=X,y=y,cv=cv)
print(np.mean(cv_scores))

predicted = cross_val_predict(GaussianNB(), X=X, y=y,  cv=cv)

print(confusion_matrix(y, predicted))
print("accuracy: " + str(accuracy_score(y, predicted)))
print(classification_report(y, predicted))

# Manera 2
print("--- Manera 2 ---")

X_train = train.drop("readmitted", axis=1).as_matrix()
y_train = train["readmitted"].as_matrix()
X_test = test.drop("readmitted", axis=1).as_matrix()
y_test = test["readmitted"].as_matrix()

def filterp(th, ProbClass1):
    """ Given a treshold "th" and a set of probabilies of belonging to class 1 "ProbClass1", return predictions """
    y = np.zeros(ProbClass1.shape[0])
    for i, v in enumerate(ProbClass1):
        if ProbClass1[i] > th:
            y[i] = 1
    return y


clf = GaussianNB()
lth = []

# We do a 20 fold crossvalidation with 10 iterations
kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_train, y_train):
    X_train2, X_test2 = X_train[train_index], X_train[test_index]
    y_train2, y_test2 = y_train[train_index], y_train[test_index]

    # Train with the training data of the iteration
    clf.fit(X_train2, y_train2)
    # Obtaining probability predictions for test data of the iteration
    probs = clf.predict_proba(X_test2)
    # Collect probabilities of belonging to class 1
    ProbClass1 = probs[:, 1]
    # Sort probabilities and generate pairs (threshold, f1-for-that-threshold)
    res = np.array([[th, f1_score(y_test2, filterp(th, ProbClass1), pos_label=1)] for th in np.sort(ProbClass1)])

    # Uncomment the following lines if you want to plot at each iteration how f1-score evolves increasing the threshold
    # plt.plot(res[:,0],res[:,1])
    # plt.show()

    # Find the threshold that has maximum value of f1-score
    maxF = np.max(res[:, 1])
    pl = np.argmax(res[:, 1])
    optimal_th = res[pl, 0]

    # Store the optimal threshold found for the current iteration
    lth.append(optimal_th)

# Compute the average threshold for all 10 iterations
thdef = np.mean(lth)
print("Selected threshold in 10-fold cross validation:", thdef)
print()

# Train a classifier with the whole training data
clf = GaussianNB()
clf.fit(X_train, y_train)
# Obtain probabilities for data on test set
probs = clf.predict_proba(X_test)

# Generate predictions using probabilities and threshold found on 10 folds cross-validation
pred = filterp(thdef, probs[:, 1])
print("---------")
print("threshold: " + str(thdef))
# Print results with this prediction vector
print("accuracy: " + str(accuracy_score(y_test, pred)))
print(classification_report(y_test, pred))
# Ignore warnings explaining that in some iterations f1 score is 0
