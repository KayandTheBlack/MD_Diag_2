import numpy as np    # Numeric and matrix computation
import pandas as pd   # Optional: good package for manipulating data
import sklearn as sk  # Package with learning algorithms implemented
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn import over_sampling as os
from sklearn.preprocessing import OneHotEncoder


path="E:/Marc/Cole/Uni/7eQ/MD/DataMiningOverDiabetics/processed_data.csv"
df=pd.read_csv(path)


df = df.drop(df.columns[0], axis=1)

df = df.drop(['X.1', 'X', 'encounter_id', 'patient_n', 'weight', 'payer_code', 'specialty'], axis=1)

drop_Idx = set(df[(df['diag_2'] == '?') & (df['diag_3'] == '?')].index)
drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
new_Idx = list(set(df.index) - set(drop_Idx))
df = df.iloc[new_Idx]

df['readmitted'] = df['readmitted'].replace('>30', 0)
df['readmitted'] = df['readmitted'].replace('<30', 1)
df['readmitted'] = df['readmitted'].replace('NO', 0)

print(df.head())

df['age'] = df['age'].replace('[0-10)', 0)
df['age'] = df['age'].replace('[10-20)', 1)
df['age'] = df['age'].replace('[20-30)', 2)
df['age'] = df['age'].replace('[30-40)', 3)
df['age'] = df['age'].replace('[40-50)', 4)
df['age'] = df['age'].replace('[50-60)', 5)
df['age'] = df['age'].replace('[60-70)', 6)
df['age'] = df['age'].replace('[70-80)', 7)
df['age'] = df['age'].replace('[80-90)', 8)
df['age'] = df['age'].replace('[90-100)', 9)

df['disch_id'] = df['disch_id'].replace(6,1)
df['disch_id'] = df['disch_id'].replace(8,1)
df['disch_id'] = df['disch_id'].replace(9,1)
df['disch_id'] = df['disch_id'].replace(13,1)
df['disch_id'] = df['disch_id'].replace(3,2)
df['disch_id'] = df['disch_id'].replace(4,2)
df['disch_id'] = df['disch_id'].replace(5,2)
df['disch_id'] = df['disch_id'].replace(14,2)
df['disch_id'] = df['disch_id'].replace(22,2)
df['disch_id'] = df['disch_id'].replace(23,2)
df['disch_id'] = df['disch_id'].replace(24,2)
df['disch_id'] = df['disch_id'].replace(12,10)
df['disch_id'] = df['disch_id'].replace(15,10)
df['disch_id'] = df['disch_id'].replace(16,10)
df['disch_id'] = df['disch_id'].replace(17,10)
df['disch_id'] = df['disch_id'].replace(25,18)
df['disch_id'] = df['disch_id'].replace(26,18)
df["disch_id"] = df["disch_id"].replace(11, np.NaN)
df.dropna(inplace = True)

df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)
df['A1Cresult'] = df['A1Cresult'].replace('None', -99)

df['circulatory'] = 0
df.loc[df['diag_1'] == 'Circulatory', 'circulatory'] = 1
df.loc[df['diag_2'] == 'Circulatory', 'circulatory'] = 1
df.loc[df['diag_3'] == 'Circulatory', 'circulatory'] = 1

df['diabetes'] = 0
df.loc[df['diag_1'] == 'Diabetes', 'diabetes'] = 1
df.loc[df['diag_2'] == 'Diabetes', 'diabetes'] = 1
df.loc[df['diag_3'] == 'Diabetes', 'diabetes'] = 1

df['digestive'] = 0
df.loc[df['diag_1'] == 'Digestive', 'digestive'] = 1
df.loc[df['diag_2'] == 'Digestive', 'digestive'] = 1
df.loc[df['diag_3'] == 'Digestive', 'digestive'] = 1

df['genitourinary'] = 0
df.loc[df['diag_1'] == 'Genitourinary', 'genitourinary'] = 1
df.loc[df['diag_2'] == 'Genitourinary', 'genitourinary'] = 1
df.loc[df['diag_3'] == 'Genitourinary', 'genitourinary'] = 1

df['injury'] = 0
df.loc[df['diag_1'] == 'Injury', 'injury'] = 1
df.loc[df['diag_2'] == 'Injury', 'injury'] = 1
df.loc[df['diag_3'] == 'Injury', 'injury'] = 1

df['musculoskeletal'] = 0
df.loc[df['diag_1'] == 'Musculoskeletal', 'musculoskeletal'] = 1
df.loc[df['diag_2'] == 'Musculoskeletal', 'musculoskeletal'] = 1
df.loc[df['diag_3'] == 'Musculoskeletal', 'musculoskeletal'] = 1

df['neoplasms'] = 0
df.loc[df['diag_1'] == 'Neoplasms', 'neoplasms'] = 1
df.loc[df['diag_2'] == 'Neoplasms', 'neoplasms'] = 1
df.loc[df['diag_3'] == 'Neoplasms', 'neoplasms'] = 1

df['other'] = 0
df.loc[df['diag_1'] == 'Other', 'other'] = 1
df.loc[df['diag_2'] == 'Other', 'other'] = 1
df.loc[df['diag_3'] == 'Other', 'other'] = 1

df['respiratory'] = 0
df.loc[df['diag_1'] == 'Respiratory', 'respiratory'] = 1
df.loc[df['diag_2'] == 'Respiratory', 'respiratory'] = 1
df.loc[df['diag_3'] == 'Respiratory', 'respiratory'] = 1

df = df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)

# stacked = df[['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'insulin', 'change', 'diabetesMed', 'other_meds']].stack()
# df[['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'insulin', 'change', 'diabetesMed', 'other_meds']] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()


categoricals = ['gender', 'race', 'metformin', 'insulin', 'change', 'diabetesMed', 'other_meds']
le = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
for vname in categoricals:
    integer_encoded = le.fit_transform(df[vname])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df[vname] = onehot_encoded

print(df.head().T)

numericV = ['age', 'time_in_hpt', 'n_lab_proc', 'n_proc', 'n_med', 'n_outp', 'n_emerg', 'n_inp', 'n_diag']
for var in numericV:
    if (abs(df[var].skew()) >2) & (abs(df[var].kurtosis()) >2):
        print(var, " needs log")
        df[var] =  np.log1p(df[var])

inputSet = df.drop("readmitted", axis=1)
outputSet = df["readmitted"]

y = df['readmitted'].values

X_train, X_test, y_train, y_test = train_test_split(inputSet, outputSet, random_state=42,test_size=0.3)

smoter = os.SMOTE(random_state = 42)
X_train, y_train = smoter.fit_sample(X_train, y_train)

TRAIN = pd.DataFrame(X_train, columns = list(inputSet))
TRAIN["TARGET"] = y_train
TRAIN.to_csv('./TRAIN.csv')

TEST = X_test
TEST["TARGET"] = y_test
TEST.to_csv('./TEST.csv')

X = df.drop(['readmitted'], axis=1).as_matrix()

lrf=[]
# for nest in [1,2,5,10,20,50,100,200]:
#     scores = cross_val_score(RandomForestClassifier(n_estimators=nest), X, y, cv=cv, scoring='accuracy')
#     print("Accuracy: %0.3f [%s]" % (scores.mean(), nest))
#     lrf.append(scores.mean())

# predicted = cross_val_predict(RandomForestClassifier(n_estimators=30), X,y, cv=50)
# print("-------")
# print(confusion_matrix(y, predicted))
# print("-------")
# print("accuracy = " + str(accuracy_score(y, predicted)))
# print("-------")
# print(classification_report(y,predicted))