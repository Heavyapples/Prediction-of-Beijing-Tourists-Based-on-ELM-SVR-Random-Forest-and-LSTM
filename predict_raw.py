import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ELM import ELM
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# 读取数据
data = pd.read_excel('preprocessed_data.xlsx')

# 去除年份字符串中的非数字字符
data['年份'] = data['年份'].str.replace('年', '')

# 转换年份数据类型
data['年份'] = data['年份'].astype(int)

# 分离特征和目标变量
X = data.drop(['年份', '接待旅游者总人数（万人）'], axis=1)
y = data['接待旅游者总人数（万人）']

# 划分训练集和测试集
train_mask = data['年份'] < 2018
test_mask = data['年份'] >= 2018
X_train, X_test, y_train, y_test = X[train_mask], X[test_mask], y[train_mask], y[test_mask]

# 加载scaler
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# 极限学习机
elm = ELM(random_state=0)
elm.fit(X_train, y_train)
y_pred_all_elm = elm.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_elm_original = scaler_y.inverse_transform(y_pred_all_elm.reshape(-1, 1))
y_actual = scaler_y.inverse_transform(y.values.reshape(-1, 1))

print('极限学习机MSE:', mean_squared_error(y_actual, y_pred_all_elm_original))

# 支持向量机
parameters = {'C': [0.5, 0.7, 0.9, 1.0], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.05, 0.1, 0.2]}
svr = SVR()
grid_svr = GridSearchCV(svr, parameters, verbose=1, scoring='neg_mean_squared_error')
grid_svr.fit(X_train, y_train)
print('Best Support Vector Regression parameters:', grid_svr.best_params_)
y_pred_all_svr = grid_svr.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_svr_original = np.expm1(scaler_y.inverse_transform(y_pred_all_svr.reshape(-1, 1)))
print('支持向量机MSE:', mean_squared_error(y_actual, y_pred_all_svr_original))

# 随机森林
parameters = {'n_estimators': [250, 300, 400], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]}
rf = RandomForestRegressor(random_state=0)
grid_rf = GridSearchCV(rf, parameters, verbose=1, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)
print('Best Random Forest parameters:', grid_rf.best_params_)
y_pred_all_rf = grid_rf.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_rf_original = np.expm1(scaler_y.inverse_transform(y_pred_all_rf.reshape(-1, 1)))
print('随机森林MSE:', mean_squared_error(y_actual, y_pred_all_rf_original))
