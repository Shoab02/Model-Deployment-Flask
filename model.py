

import sklearn
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

#loading the data
data = sklearn.datasets.load_iris()

#test and train split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.8)
print(X_train, y_train)

#Train the model
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)

#Test the model
result = model.score(X_test, y_test)
print(result)

#save the model
filename = 'iris-model.pkl'
pickle.dump(model, open(filename, 'wb'))

