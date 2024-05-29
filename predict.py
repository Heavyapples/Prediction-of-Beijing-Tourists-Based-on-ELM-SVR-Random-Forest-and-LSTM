import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ELM import ELM
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取数据
data = pd.read_excel('preprocessed_data_PCA.xlsx')

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
y_pred_all = elm.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_original = scaler_y.inverse_transform(y_pred_all.reshape(-1, 1))
y_actual = scaler_y.inverse_transform(y.values.reshape(-1, 1))

print('极限学习机MSE:', mean_squared_error(y_actual, y_pred_all_original))

# 绘制所有年份的预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], y_pred_all_original.flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], y_actual.flatten(), 'b-', label='Actual')
plt.title('极限学习机预测结果')
plt.xlabel('年份')
plt.ylabel('接待旅游者总人数（万人）')
plt.legend()
plt.show()

# 支持向量回归
svr = SVR(C=1, gamma='auto', epsilon=0.1).fit(X_train, y_train)
y_pred_all_svr = svr.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_svr_original = np.expm1(scaler_y.inverse_transform(y_pred_all_svr.reshape(-1, 1)))
y_actual = np.expm1(scaler_y.inverse_transform(y.values.reshape(-1, 1)))
print('支持向量机MSE:', mean_squared_error(y_actual, y_pred_all_svr_original))

# 绘制所有年份的预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], y_pred_all_svr_original.flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], y_actual.flatten(), 'b-', label='Actual')
plt.title('支持向量机预测结果')
plt.xlabel('年份')
plt.ylabel('接待旅游者总人数（万人）')
plt.legend()
plt.show()

# 随机森林
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rf.fit(X_train, y_train)
y_pred_all_rf = rf.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_rf_original = np.expm1(scaler_y.inverse_transform(y_pred_all_rf.reshape(-1, 1)))
print('随机森林MSE:', mean_squared_error(y_actual, y_pred_all_rf_original))

# 绘制所有年份的预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], y_pred_all_rf_original.flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], y_actual.flatten(), 'b-', label='Actual')
plt.title('随机森林预测结果')
plt.xlabel('年份')
plt.ylabel('接待旅游者总人数（万人）')
plt.legend()
plt.show()

# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_all_lr = lr.predict(X)

# 将预测结果还原为原始的尺度
y_pred_all_lr_original = np.expm1(scaler_y.inverse_transform(y_pred_all_lr.reshape(-1, 1)))
print('线性回归MSE:', mean_squared_error(y_actual, y_pred_all_lr_original))

# 绘制所有年份的预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], y_pred_all_lr_original.flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], y_actual.flatten(), 'b-', label='Actual')
plt.title('线性回归预测结果')
plt.xlabel('年份')
plt.ylabel('接待旅游者总人数（万人）')
plt.legend()
plt.show()

# 为LSTM模型准备数据
X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
X_lstm = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))

# 创建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练LSTM模型
model.fit(X_train_lstm, y_train, epochs=50, batch_size=1, verbose=2)

# 使用LSTM模型进行预测
y_pred_all_lstm = model.predict(X_lstm)

# 将预测结果还原为原始的尺度
y_pred_all_lstm_original = np.expm1(scaler_y.inverse_transform(y_pred_all_lstm))
print('LSTM MSE:', mean_squared_error(y_actual, y_pred_all_lstm_original))

# 绘制所有年份的预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], y_pred_all_lstm_original.flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], y_actual.flatten(), 'b-', label='Actual')
plt.title('LSTM预测结果')
plt.xlabel('年份')
plt.ylabel('接待旅游者总人数（万人）')
plt.legend()
plt.show()
