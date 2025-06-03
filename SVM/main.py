#diabetes prediction model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

data= pd.read_csv('/content/diabetes.csv')

data.head()

data['Outcome'].value_counts()

data.groupby('Outcome').mean()

x= data.drop(columns='Outcome', axis=1)
y=data['Outcome']
print(x)
print(y)

scaler= StandardScaler()
scaler.fit(x)
stand_data=scaler.transform(x)
print(stand_data)

X = stand_data
Y = data['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)

classifier= svm.SVC(kernel='linear')

classifier.fit(x_train,y_train)

x_train_prediction=classifier.predict(x_train)
training_data_acc= accuracy_score(x_train_prediction, y_train)
print(training_data_acc)

x_test_prediction= classifier.predict(x_test)
testing_data_acc=accuracy_score(x_test_prediction,y_test)
print(testing_data_acc)

input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
