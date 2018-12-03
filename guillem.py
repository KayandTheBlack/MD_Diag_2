
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd   # Optional: good package for manipulating data 
import numpy as np

" Importing the data"
df = pd.read_csv("processed_data.csv",index_col=0)

"Saving a list of all columns names without the id ones"
allColumnNames = df.columns.values.copy()[2:]
#ids = ["encounter_id","patient_n"]

"Columns classified in lists depending on their type"
numerical = ["weight", "time_in_hpt","n_lab_proc","n_proc","n_med","n_outp",
             "e_emerg","n_inp","n_diag"]
categoricals = ['race', 'age', #'gender',
       'adm_type_id', 'disch_id', 'adm_source_id',
       'payer_code', 'specialty',
       'diag_1', 'diag_2', 'diag_3',
       'A1Cresult', 'metformin', 'insulin']#, 'change',
       #'diabetesMed', 'other_meds'] # without the label one

"Dropping the id-like columns from the dataset"
df = df.drop('encounter_id',axis=1)
df = df.drop('patient_n',axis=1)

"Making sure categorical columns are treated as such"
for col in categoricals:
    df[col] = df[col].astype('category')

"Transforming binary columns into a more appropiate format"
df['diabetesMed'] = df['diabetesMed'].replace({'Yes':1,'No':0})
df['other_meds'] = df['other_meds'].replace({'takes_more_meds':1,'no_more_meds':0})
df['change'] = df['change'].replace({'Ch':1,'No':0})
df['gender'] = df['gender'].replace({'Male':1,'Female':0})

df = df.rename({'gender':'male'}, axis='columns')

df['diabetesMed'] = df['diabetesMed'].astype('bool')
df['other_meds'] = df['other_meds'].astype('bool')
df['change'] = df['change'].astype('bool')
df['male'] = df['male'].astype('bool')

#dataColumnsNames = np.delete(allColumnNames, np.where(allColumnNames == 'readmitted'), axis=0) 

"Substituting the columns consisting of ranges with their averages"
weightAvgDict = {"[75-100)" : 87.5, "[50-75)" : 62.5, "[0-25)":12.5,
                 "[100-125)":112.5, "[25-50)":37.5, "[125-150)":137.5,
                 "[175-200)":187.5, "[150-175)":162.5, ">200":212.5}

df['weight'] = df['weight'].replace(weightAvgDict)

ageAvgDict = {"[50-60)":55 , "[80-90)":85, "[60-70)":65, "[40-50)":45,
              "[70-80)":75, "[30-40)":35, "[0-10)":5, "[90-100)":95,
              "[10-20)":15, "[20-30)":25}
df['age'] = df['age'].replace(ageAvgDict)


"Transforming the target variable form trinary to binary"
df['readmitted'] = df['readmitted'].replace({'>30':'NO'})



'''
# calculate correlation matrix and store as absolute values
c = df.corr().abs()

# unstake the table
s = c.unstack()

# sort the values in descending order
so = s.sort_values(ascending=False)
print(so[14:60])
'''

'''Trying things with covariance, commented out since it didn't seem to
affect results'''
interactionterms = [("diabetesMed",  "other_meds"),
                    ("change",       "other_meds"),
                    ("time_in_hpt",  "n_med"),
                    ("change",       "diabetesMed"),
                    ("n_med",        "n_proc"),
                    ("n_inp",        "n_emerg"),
                    ("time_in_hpt",  "n_lab_proc"),
                    ("n_diag",       "age")]
'''
for inter in interactionterms:
    name = inter[0] + '|' + inter[1]
    df[name] = df[inter[0]] * df[inter[1]]
'''

#df['service_utilization'] = df['n_outp'] + df['n_emerg'] + df['n_inp']


"Imputation of unknown race"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le =le.fit(df['race'])
list(le.classes_)
df['race'] = le.transform(df['race'])

#df['race'] = le.fit_transform(df['race'])

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=4, strategy='median', axis=0)
imp.fit(df[['race']])
df['race'] = imp.transform(df[['race']]).ravel()

df['race'] = df['race'].astype('int64')

df['race'] = le.inverse_transform(df['race'])



def labelEncode():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for i in categoricals:
        df[i] = le.fit_transform(df[i])


# Get one hot encoding of columns B
# Drop column B as it is now encoded
# Join the encoded df

def onehotEncode():
    global categoricals
    global df
    for i in categoricals:
        one_hot = pd.get_dummies(df[i])
        df = df.drop(i,axis=1)
        df = df.join(one_hot, lsuffix=i)


"Applying one-hot encoding to all categorical variables"
onehotEncode()
    
#realy=df['readmitted']
#realx=df.drop('readmitted',axis=1)

"Splitting data into training and testing"
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
train, test = train_test_split(df, test_size=0.3)
#thing = KFold(n_splits=3,shuffle=True,random_state=123)

'''
## DownSampling
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = train[train.readmitted=='NO']
df_minority = train[train.readmitted!='NO']
 
import random
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=int(len(df_minority)*1),     # to match minority class
                                 random_state=random.randint(1000,100000)) # reproducible results with 123, prec 60ish, rec 70ish
# uncomment to turn on downsampling
#train = pd.concat([df_majority_downsampled, df_minority])
'''

"Separating the data sets into data and target"
y_train=train['readmitted']
X_train=train.drop('readmitted',axis=1)
y_val=test['readmitted']
X_val=test.drop('readmitted',axis=1)


"----------"
dec = 0 #default is 0 a good value seemed 0.000005

"Regular classifier"
clf = RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec)#,criterion = 'entropy')
clf = clf.fit(X_train, y_train)
#print(clf.feature_importances_)  


from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import cross_val_score

"With feature selection"
model = SelectFromModel(clf, prefit=True, threshold=0.01)
X_trainth = model.transform(X_train)
X_valth = model.transform(X_val)

clfth = RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec)#,criterion = 'entropy')
clfth = clfth.fit(X_trainth, y_train)
#print(X_train_new.shape)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''Get the predictions of the classifier and the matrix of probabilities of
each class for each row'''
pred = clf.predict(X_val)
probs = clf.predict_proba(X_val)

predth=clfth.predict(X_valth)
probsth = clfth.predict_proba(X_valth)

"Making a custom prediction with a much lower threshold"
th = 0.09
count = 0
customPred = [""]*len(pred)

for i in range(0,len(pred)):
    if (probs[i][0] < th):
        customPred[i] = 'NO'
    else:
        customPred[i] = '<30'

for i in range(0,len(pred)):
    if (pred[i] == customPred[i]):
        count += 1
        
print(len(pred))
print(count)

"Same but for the one with feature selection"
count = 0
customPredth = [""]*len(predth)

for i in range(0,len(predth)):
    if (probsth[i][0] < th):
        customPredth[i] = 'NO'
    else:
        customPredth[i] = '<30'

for i in range(0,len(predth)):
    if (predth[i] == customPredth[i]):
        count += 1
        
print(len(predth))
print(count)

print("\n No feature selection, original vs custom \n")
rep = classification_report(y_val, pred)
customRep = classification_report(y_val, customPred)

print(rep)
print(customRep)

confus = confusion_matrix(y_val, pred)
customConfus = confusion_matrix(y_val, customPred)

print(confus)
print(customConfus)

print("\n\n With feature selection, original vs custom \n")
repth = classification_report(y_val, predth)
customRepth = classification_report(y_val, customPredth)

print(repth)
print(customRepth)

confusth = confusion_matrix(y_val, predth)
customConfusth = confusion_matrix(y_val, customPredth)

print(confusth)
print(customConfusth)

#print(np.mean(cross_val_score(clf, X=realx, y=realy, cv=30, scoring='accuracy')))
#print(np.mean(cross_val_score(clf, X=realx, y=realy, cv=30, scoring='accuracy')))




'''
def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
#    dataframe = pd.DataFrame.from_dict(report_data)
    return report_data
'''

#report = [[0,0],[0,0]]
#report += classification_report_csv(rep)
#
#

#[TP, FN] = confus[0]
#[FP, TN] = confus[1]


#print(np.mean(cross_val_score(clf, X=realx, y=realy, cv=30, scoring='accuracy')))
 




'''
from imblearn.over_sampling import SMOTE
from collections import Counter
#print('Original dataset shape {}'.format(Counter(y_train)))
sm = SMOTE(random_state=20)
X_train2, y_train2 = sm.fit_sample(X_train, y_train)
#print('New dataset shape {}'.format(Counter(y_train2)))

clf = RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec)#,criterion = 'entropy')
clf = clf.fit(X_train2, y_train2)
#print(clf.feature_importances_)  


model = SelectFromModel(clf, prefit=True, threshold=0.005)
X_train_new2 = model.transform(X_train2)
X_val_new2 = model.transform(X_val)
print(X_train_new2.shape)


pred=clf.predict(X_val)
print(classification_report(y_val, pred))

clf = RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec)
clf = clf.fit(X_train_new2, y_train2)
pred=clf.predict(X_val_new2)
print(classification_report(y_val, pred))
'''

#cv=30
#print(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec), X=X_train, y=y_train, cv=cv, scoring='accuracy')))
#
#print(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec), X=X_train_new, y=y_train, cv=cv, scoring='accuracy')))
#
#print(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec), X=X_train2, y=y_train2, cv=cv, scoring='accuracy')))
#
#print(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100,min_impurity_decrease=dec), X=X_train_new2, y=y_train2, cv=cv, scoring='accuracy')))

