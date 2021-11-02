import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
import sklearn

# read
df = pd.read_csv('data/student-mat.csv', sep=';')

df['qual_student'] = np.where(df['G3']>=15, 1, 0)
# print(df.describe())

all_labels = [
    'school',       # - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
    'sex',          # - student's sex (binary: "F" - female or "M" - male)
    'age',          # - student's age (numeric: from 15 to 22)
    'address',      # - student's home address type (binary: "U" - urban or "R" - rural)
    'famsize',      # - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
    'Pstatus',      # - parent's cohabitation status (binary: "T" - living together or "A" - apart)
    'Medu',         # - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    'Fedu',         # - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    'Mjob',         # - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
    'Fjob',         # - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
    'reason',       # - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
    'guardian',     # - student's guardian (nominal: "mother", "father" or "other")
    'traveltime',   # - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
    'studytime',    # - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    'failures',     # - number of past class failures (numeric: n if 1<=n<3, else 4)
    'schoolsup',    # - extra educational support (binary: yes or no)
    'famsup',       # - family educational support (binary: yes or no)
    'paid',         # - extra paid classes within the course subject (binary: yes or no)
    'activities',   # - extra-curricular activities (binary: yes or no)
    'nursery',      # - attended nursery school (binary: yes or no)
    'higher',       # - wants to take higher education (binary: yes or no)
    'internet',     # - Internet access at home (binary: yes or no)
    'romantic',     # - with a romantic relationship (binary: yes or no)
    'famrel',       # - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
    'freetime',     # - free time after school (numeric: from 1 - very low to 5 - very high)
    'goout',        # - going out with friends (numeric: from 1 - very low to 5 - very high)
    'Dalc',         # - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    'Walc',         # - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    'health',       # - current health status (numeric: from 1 - very bad to 5 - very good)
    'absences',     # - number of school absences (numeric: from 0 to 93)
]

# include = [
#     'school',       # - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
#     'age',          # - student's age (numeric: from 15 to 22)
#     'Medu',         # - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
#     'Fedu',         # - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
#     'reason',       # - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
#     'traveltime',   # - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
#     'studytime',    # - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
#     'failures',     # - number of past class failures (numeric: n if 1<=n<3, else 4)
#     'schoolsup',    # - extra educational support (binary: yes or no)
#     'famsup',       # - family educational support (binary: yes or no)
#     'paid',         # - extra paid classes within the course subject (binary: yes or no)
#     'activities',   # - extra-curricular activities (binary: yes or no)
#     'higher',       # - wants to take higher education (binary: yes or no)
#     'internet',     # - Internet access at home (binary: yes or no)
#     'romantic',     # - with a romantic relationship (binary: yes or no)
#     'famrel',       # - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
#     'freetime',     # - free time after school (numeric: from 1 - very low to 5 - very high)
#     'goout',        # - going out with friends (numeric: from 1 - very low to 5 - very high)
#     'Dalc',         # - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
#     'Walc',         # - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
#     'health',       # - current health status (numeric: from 1 - very bad to 5 - very good)
#     'absences',     # - number of school absences (numeric: from 0 to 93)
# ]

core = [
    'failures',     # - number of past class failures (numeric: n if 1<=n<3, else 4)
    'absences',     # - number of school absences (numeric: from 0 to 93)
    'school',
]

while True:
    highest_f1 = 0
    highest_label = ''
    
    # base
    include = core.copy()
    include.append('qual_student')
    df_ = df[include].copy()

    dependent_variable = 'qual_student'
    df_ = pd.get_dummies(df_)

    X = df_[df_.columns.difference([dependent_variable])]
    y = df_[dependent_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    base_f1_score = 0
    base_precision_score = 0
    base_recall_score = 0
    for i in range(10):
        clf = rf(n_estimators = 100)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        # print(sklearn.metrics.f1_score(y_train, clf.predict(X_train), average='binary'))
        base_f1_score += sklearn.metrics.f1_score(y_test, pred, average='binary')
        base_precision_score += sklearn.metrics.precision_score(y_test, pred, average='binary')
        base_recall_score += sklearn.metrics.recall_score(y_test, pred, average='binary')
    print(core)
    print("f1 score: %.4f" % (base_f1_score / 10))

    for label in [item for item in all_labels if item not in core]:
        include = core.copy()
        include.append(label)
        include.append('qual_student')
        df_ = df[include].copy()

        dependent_variable = 'qual_student'
        df_ = pd.get_dummies(df_)

        X = df_[df_.columns.difference([dependent_variable])]
        y = df_[dependent_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # test fit 10 times
        total_f1_score = 0
        total_precision_score = 0
        total_recall_score = 0
        for i in range(10):
            clf = rf(n_estimators = 100)
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            total_f1_score += sklearn.metrics.f1_score(y_test, pred, average='binary')
            total_precision_score += sklearn.metrics.precision_score(y_test, pred, average='binary')
            total_recall_score += sklearn.metrics.recall_score(y_test, pred, average='binary')
        f1_improvement = total_f1_score / 10 - base_f1_score / 10
        # print("-----%s-----" % label)
        # print("f1 score: %.4f" % (total_f1_score / 10 - base_f1_score / 10))
        # print("precision score: %.4f" % (total_precision_score / 10 - base_precision_score / 10))
        # print("recall score: %.4f" % (total_recall_score / 10 - base_precision_score / 10))
        if f1_improvement > highest_f1:
            highest_f1 = f1_improvement
            highest_label = label
    if (highest_f1 <= 0):
        print(core)
        exit(0)
    print("%s improved score by %.4f" % (highest_label, highest_f1))
    core.append(highest_label)