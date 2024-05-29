import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取数据
data_scaled = pd.read_excel('preprocessed_data.xlsx')

# 设置要进行PCA的特征
features = data_scaled.columns.drop(['年份', '接待旅游者总人数（万人）'])

# 进行PCA
pca = PCA()
principalComponents = pca.fit_transform(data_scaled[features])

# 绘制解释方差比例图
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('主成分数量')
plt.ylabel('方差解释比例（%）')
plt.title('方差解释比例')
plt.show()

# 根据解释方差比例图选择合适数量的主成分，例如选择解释方差比例达到95%的主成分数
n_components = np.where(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]
print('解释95%方差所需的主成分数量为：', n_components)

pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(data_scaled[features])
principalDf = pd.DataFrame(data=principalComponents, columns=['主成分' + str(i) for i in range(1, n_components+1)])

# 把PCA处理过的特征和目标变量连在一起，然后保存到新的Excel文件中
data_PCA = pd.concat([data_scaled[['年份', '接待旅游者总人数（万人）']], principalDf], axis=1)
data_PCA.to_excel('preprocessed_data_PCA.xlsx', index=False)

# 进行递归特征消除 (RFE)
X = principalDf
y = data_scaled['接待旅游者总人数（万人）']

estimator = LinearRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)

# 输出选择的特征
print('最佳特征数量为 {}'.format(selector.n_features_))
features = [f for f, s in zip(X.columns, selector.support_) if s]
print('选择的特征为：')
print(features)
