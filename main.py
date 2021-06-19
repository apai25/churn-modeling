import numpy as np 
import pandas as pd 

dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X_train[:, [0, 3, 4, 5, 6, 9]] = X_scale.fit_transform(X_train[:, [0, 3, 4, 5, 6, 9]])
X_test[:, [0, 3, 4, 5, 6, 9]] = X_scale.transform(X_test[:, [0, 3, 4, 5, 6, 9]])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
geography_encoder = OneHotEncoder(dtype=int, drop='first')
gender_encoder = OneHotEncoder(dtype=int, drop='first')
credit_card_encoder = OneHotEncoder(dtype=int, drop='first')
active_encoder = OneHotEncoder(dtype=int, drop='first')

ct = ColumnTransformer([('geography', geography_encoder, [1]), ('gender', gender_encoder, [2]), ('HasCrCard', credit_card_encoder, [7]), 
                      ('IsActiveMember', active_encoder, [8])], remainder='passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

X_train=np.asarray(X_train).astype(np.float32)
X_test=np.asarray(X_test).astype(np.float32)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(units=32, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)

model.save('model')

predictions = model.predict(X_test)
predictions = (predictions > 0.5)

from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(cm)
print(f1)