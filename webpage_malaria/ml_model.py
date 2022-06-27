import pandas as pd
#import numpy as np
import pickle
from sklearn import preprocessing

df = pd.read_csv("outbreak_detect.csv")

df=df.dropna()

#data processing

#labelencoding
LE=preprocessing.LabelEncoder()
#fitting it to our dataset
df.Outbreak = LE.fit_transform(df.Outbreak)

df=df.drop(['Positive','pf'],axis=1)

#method 2 to load the data in the form of arrays -by library numpy
#X = np.array(df[['avgHumidity',	'Rainfall',	'Positive',	'pf']])
#Y = np.array(df[['Outbreak']])
X=df.iloc[:,0:4].values
Y=df.iloc[:,-1:].values
print(X)
print(Y)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)

y_pred = model.predict(sc.transform(X_test))
print(y_pred)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print("sucess loaded")