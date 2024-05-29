import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取预处理后的数据
data_scaled = pd.read_excel('preprocessed_data.xlsx')

# 将年份中的"年"字去掉，转为日期格式并设置为索引
data_scaled['年份'] = data_scaled['年份'].str.replace('年', '')
data_scaled['年份'] = pd.to_datetime(data_scaled['年份'], format='%Y')
data_scaled.set_index('年份', inplace=True)

# 描述性统计分析
print(data_scaled.describe())

# 相关性分析
correlation = data_scaled.corr()
print(correlation)

# 可视化相关性矩阵
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.show()

# 直方图，查看各变量分布情况
data_scaled.hist(bins=50, figsize=(20,15))
plt.show()

# 散点图，查看“接待旅游者总人数（万人）”与其他变量的关系
for column in data_scaled.drop("接待旅游者总人数（万人）", axis=1).columns:
    data_scaled.plot(kind="scatter", x=column, y="接待旅游者总人数（万人）")
    plt.show()
