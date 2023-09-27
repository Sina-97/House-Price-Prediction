from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

data =pd.read_csv('train.csv')
df = pd.DataFrame(data)

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])


corr = df.corr()
needed_columns = corr['SalePrice'].sort_values(ascending=False)[:27].index.tolist()

data_features = df[needed_columns]

data_features = data_features.fillna(data_features.median())

data_train,data_test = train_test_split(data_features,test_size=0.25,random_state=111)

X_train = data_train.drop(['SalePrice'],axis=1)
y_train = data_train['SalePrice']
X_train=preprocessing.minmax_scale(X_train)


X_test = data_test.drop(['SalePrice'],axis=1)
y_test = data_test['SalePrice']
X_test=preprocessing.minmax_scale(X_test)
  

lr=LinearRegression(fit_intercept=True, n_jobs=2)
rfr=RandomForestRegressor(random_state=1,n_estimators=260,max_depth=10,max_features=22,bootstrap=True,max_leaf_nodes=1000,criterion='squared_error',min_impurity_decrease=1)
svm=SVR(kernel='rbf',C=11000,gamma=0.12)
xgb=XGBRegressor(tree_method='gpu_hist',n_estimators=1000,max_depth=10,learning_rate=0.005,alpha=20,colsample_bytree=0.7)
    

#Train the model using the training sets
model_rfr = rfr.fit(X_train, y_train)
model_lr=lr.fit(X_train, y_train)
model_svm=svm.fit(X_train, y_train)
model_xgb=xgb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_rfr = model_rfr.predict(X_test)
y_pred_lr=model_lr.predict(X_test)
y_pred_svm=model_svm.predict(X_test)
y_pred_xgb=model_xgb.predict(X_test)

score_rfr = model_rfr.score(X_test, y_test)
score_lr=model_lr.score(X_test, y_test)
score_svm=model_svm.score(X_test, y_test)
score_xgb=model_xgb.score(X_test, y_test)


mse_rfr = np.mean((y_pred_rfr - y_test)**2)
mse_lr = np.mean((y_pred_lr - y_test)**2)
mse_svm = np.mean((y_pred_svm - y_test)**2)
mse_xgb = np.mean((y_pred_xgb - y_test)**2)
        

print("Random Forest Regressor score is: ",score_rfr)
print("Linear Regression score is: ",score_lr)
print("SVM score is: ",score_svm)
print("XGBoost score is: ",score_xgb)
# print("Random Forest Regressor MSE is: ",mse_rfr)
# print("Linear Regression MSE is: ",mse_lr)
# print("SVM MSE is: ",mse_svm)
# print("XGBoost MSE is: ",mse_xgb)

data_final=pd.read_csv('test.csv')
df_final = pd.DataFrame(data_final)
for column in df_final.columns:
    if df_final[column].dtype == 'object':
        df_final[column] = encoder.fit_transform(df_final[column])
        
needed_columns.remove('SalePrice')
data_final_features = df_final[needed_columns]
data_final_features = data_final_features.fillna(data_final_features.median())
X_final =data_final_features
X_final=preprocessing.minmax_scale(X_final)

score_list=[score_rfr,score_lr,score_svm,score_xgb]
mse_list=[mse_rfr,mse_lr,mse_svm,mse_xgb]
model_list=[model_rfr,model_lr,model_svm,model_xgb]
model_final=model_list[score_list.index(max(score_list))]

if model_final==model_rfr:
    print("Random Forest Regressor is the best model")
elif model_final==model_lr:
    print("Linear Regression is the best model")
elif model_final==model_svm:
    print("SVM is the best model")
elif model_final==model_xgb:
    print("XGBoost is the best model")
else:
    print("Error")


y_final_pred=model_final.predict(X_final)
print('Final prediction is: ',y_final_pred)



