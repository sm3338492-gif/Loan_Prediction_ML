import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, f1_score , recall_score
data = pd.read_csv("Default_Fin.csv")
df = pd.DataFrame(data)
print(df.head())

# Checking the null value and filtering the dataset

null_value = df.isnull().sum()
print(null_value)

# Using Decision Classifier Tree for model to train on data

df['Loan_Prediction'] = ((df["Annual Salary"] >= 300000) | (df["Bank Balance"] >= 25000)).astype(int)
X = df[["Annual Salary" , "Bank Balance" , "Employed"]]
y = df["Loan_Prediction"]

# training and testing

X_train , X_test , y_train ,y_test = train_test_split(X , y ,test_size = 0.2)

#using standard scaling for balancing the data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')

model.fit(X_train ,y_train)

annual_salary = float(input("Enter Your Annual Salary :"))
bank_balance = float(input("Enter your Bank Balance :"))
employed = float(input("Are you employee ?(0/1) :"))

input_data = pd.DataFrame([[annual_salary, bank_balance, employed]],
                          columns=["Annual Salary", "Bank Balance", "Employed"])

input_scaled = scaler.transform(input_data)

user_prob = model.predict_proba(input_scaled)[:,1]
Model_Prediction = (user_prob > 0.5).astype(int)

print("Default Probability:", user_prob[0])

y_pred = model.predict(X_test)
print("Accuracy :" , accuracy_score(y_test,y_pred))
print("Precision Score :" , precision_score(y_test , y_pred))
print("Recall Score :" , recall_score(y_test ,y_pred))
print("F1 Score :", f1_score(y_test , y_pred))

print(df['Loan_Prediction'].value_counts())

if(Model_Prediction[0] == 1):
    print("Your Loan has been approved!!")
else:
    print("Your Loan hasn't been approved!!")





