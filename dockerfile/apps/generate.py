import pandas as pd
import numpy as np
df = pd.read_csv('student-mat.csv', sep=';')
df['qual_student'] = np.where(df['G3']>=15, 1, 0)
include = ['school', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'activities', 'higher', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

# by doing inclusion w/out the G3 we are able to remove G3 from it (shouldn't be considered)
include.append('qual_student')
df.drop(columns=df.columns.difference(include), inplace=True) 

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
import sklearn

# create "x" df of the features that determine the "y" df, split data into training and test
dependent_variable = 'qual_student'
df = pd.get_dummies(df) # need to change
x = df[df.columns.difference([dependent_variable])]
y = df[dependent_variable]

# benchmark the model by running 10 times to see what the average scores are
accuracy_sum = 0.0
f1_sum = 0.0
runs = 50
for i in range(runs):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80/20 train/test split
    clf = rf(n_estimators=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy_sum += sklearn.metrics.accuracy_score(y_test, y_pred)
    f1_sum += sklearn.metrics.f1_score(y_test, y_pred, average='binary')

accuracy_score = accuracy_sum / runs
f1_score = f1_sum / runs

print("Model Statistics on Test Data")
print("Accuracy: " + str(accuracy_score))
print("FScore: " + str(f1_score))