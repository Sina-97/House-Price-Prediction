from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_id = test_data["Id"]
train_data_saleprice = train_data["SalePrice"]
train_labels = train_data.pop("SalePrice")


train_data.fillna("0", inplace=True)
test_data.fillna("0", inplace=True)

encoder=LabelEncoder()

for column in train_data.select_dtypes(exclude=["number"]).columns.intersection(test_data.select_dtypes(exclude=["number"]).columns):
    encoder.fit_transform(train_data[column].to_list() + test_data[column].to_list())
    
    train_data[column] = encoder.transform(train_data[column])
    test_data[column] = encoder.transform(test_data[column])
    
train_data=train_data.astype(float)
test_data=test_data.astype(float)

scaler=preprocessing.StandardScaler()

train_data=scaler.fit_transform(train_data)
test_data=scaler.transform(test_data)

model=XGBRegressor(tree_method='gpu_hist')
xgb_parameters = {
    'n_estimators': [100,400,500],
    'max_depth': [3,4,5],
    'learning_rate': [0.1,0.01,0.001],
    "alpha": [5, 10, 15],
    "colsample_bytree": [0.1,0.3,0.5]
}

xgbr_search=GridSearchCV(model, xgb_parameters,cv=2,scoring='neg_mean_squared_error')
xgbr_search.fit(train_data, train_labels)
best_model=xgbr_search.best_estimator_
prediction=best_model.predict(test_data)

score=best_model.score(train_data, train_labels)
print("Final Score: ",score)

