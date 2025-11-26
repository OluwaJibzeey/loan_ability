import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df=pd.read_csv('loan_ability_new.csv')
print (df)
print (df.head())
print (df.columns)
print (df.dtypes)
print (df.shape)
print (df.isnull().sum())
print (df.dropna(inplace=True))
print (df.isnull().sum())
df.drop(['Gender','date'],inplace=True,axis=1)
df.head()
textcol=df[['Married','Education','Self_Employed','Area','Status']]
Label=LabelEncoder()
for col in textcol:
    df[col]=Label.fit_transform(df[col])
df['Dependents'].unique()
df['Dependents']=df['Dependents'].str.replace('+',' ').astype (int)
x=df.drop(['Status'],axis=1)
y=df['Status']
from sklearn.metrics import accuracy_score
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=(0.2),random_state=42)
RFC=RandomForestClassifier()
RFC.fit(xtrain,ytrain)
RFCPred=RFC.predict(xtest)
print (RFCPred)
RFCAccuracy=accuracy_score(RFCPred,ytest)*100
print (RFCAccuracy)
import joblib
joblib.dump(RFC, "loan model.pkl")
print("modelsaved")






