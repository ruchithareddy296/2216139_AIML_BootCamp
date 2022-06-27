import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("SUV_Purchase.csv")

df = df.drop('User ID', axis=1)
df = df.drop('Gender', axis=1)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(sc.transform(X_test))
print(y_pred)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print("sucess loaded")

#execute this file only once and create the pkl file