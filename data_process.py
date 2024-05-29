import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pickle

# 读取数据，设置header和index_col参数
data = pd.read_excel('data.xlsx', header=0, index_col=0)

# 转置数据，使其变成长格式
data = data.transpose()

# 处理缺失值，使用平均值填充
data.fillna(data.mean(), inplace=True)

# 异常值处理，这里我们使用Z-score方法，Z-score是一个统计学概念，表示数据点与平均值的距离，以标准差为单位。
# Z-score的绝对值大于3通常被认为是异常值
z_scores = np.abs(stats.zscore(data.drop(['接待旅游者总人数（万人）'], axis=1)))
filtered_entries = (z_scores < 3).all(axis=1)
data = data[filtered_entries]

# 保存年份列并转为 DataFrame
years = pd.DataFrame(data.index, columns=['年份'])

# 对数变换，加1保证所有数据为正
data_log = np.log1p(data.drop(['接待旅游者总人数（万人）'], axis=1))
data_y_log = np.log1p(data['接待旅游者总人数（万人）'])

# 数据标准化，使得每一列数据都有0均值，1标准差
scaler_X = StandardScaler()
scaler_y = StandardScaler()
data_scaled_X = pd.DataFrame(scaler_X.fit_transform(data_log), columns=data_log.columns, index=data_log.index)
data_scaled_y = pd.DataFrame(scaler_y.fit_transform(data_y_log.to_frame()), columns=['接待旅游者总人数（万人）'], index=data_y_log.index)

# 重置 years 的索引以匹配 data_scaled_X 和 data_scaled_y
years.reset_index(drop=True, inplace=True)
data_scaled_X.reset_index(drop=True, inplace=True)
data_scaled_y.reset_index(drop=True, inplace=True)

# 将年份列和目标变量列添加回去
data_scaled = pd.concat([years, data_scaled_X, data_scaled_y], axis=1)

# 保存预处理后的数据到Excel文件
data_scaled.to_excel('preprocessed_data.xlsx', index=False)

# 保存scaler到文件
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# 打印处理后的数据
print(data_scaled)
