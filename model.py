# Importing Libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load

# importing cleaned data.
training_data = pd.read_csv('training_data_algorithm.csv')
training_data.drop('Unnamed: 0', axis=1, inplace=True)
print(training_data.head())

training_set_without_survived = training_data.drop("Survived", axis=1)
training_set_with_only_survived = training_data["Survived"]

# Performing Train-Test split.
X_train, X_test, y_train, y_test = train_test_split(
    training_set_without_survived, training_set_with_only_survived, train_size=0.7, test_size=0.3, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # transforming "training"
X_test = sc.transform(X_test)  # transforming "test"


def transform_prediction_data(data):  # When a user wants to predict wwe will transform those values.
    prediction_transformed = sc.transform(data)
    return prediction_transformed


# Model Creation
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
train_result = logreg.score(X_train, y_train)
test_result = logreg.score(X_test, y_test)
print("Logistic Regression Training Accuracy: %.2f%%" % (train_result * 100.0))
print('*' * 100)
print("Logistic Regression Testing Accuracy: %.2f%%" % (test_result * 100.0))
filename1 = 'logistic_model.sav'
dump(logreg, filename1)
