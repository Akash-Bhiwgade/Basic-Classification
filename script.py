import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('data.csv')

y = np.array(df['diagnosis'])
X = np.array(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1))

enc = preprocessing.LabelEncoder()
enc.fit(['M', 'B'])
y = enc.transform(y)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#clf = svm.SVC(gamma='auto')
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

prediction = clf.predict(X_test)