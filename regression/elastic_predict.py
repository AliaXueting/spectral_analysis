import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

plt.rcParams['font.sans-serif']=['SimHei']

if __name__ == "__main__":
    df = pd.read_csv("regression_data.csv", encoding='gbk')
    df = df.drop(labels='Id', axis=1)

    df1 = df.drop(labels='性状2', axis=1)
    df2 = df.drop(labels='性状1', axis=1)
    cor1 = df1.corr(method='pearson')
    cor2 = df2.corr(method='pearson')
    print(cor1)
    print(cor2)

    #相关性系数热力图
    plt.figure(figsize=(20,20))
    sns.heatmap(cor1, annot=True, vmax=1, vmin=-1,
                xticklabels=True,
                yticklabels=True, square=True, cmap="YlGnBu")
    plt.savefig("1.png")

    plt.figure(figsize=(20, 20))
    sns.heatmap(cor2, annot=True, vmax=1, vmin=-1,
                xticklabels=True,
                yticklabels=True, square=True, cmap="YlGnBu")
    plt.savefig("2.png")

    cor1_column = abs(cor2['性状2'] > 0.3)
    #print(cor1_column)
    row_index1 = cor1[cor1['性状1'].abs() > 0.3].index.tolist()
    row_index2 = cor2[cor2['性状2'].abs() > 0.3].index.tolist()
    # print(row_index1)
    # print(row_index2)

    #岭回归模型
    column1 = ['blueref', 'greenref', 'redref', 'rededgeref', 'NIRref', 'CIre', 'DVI', 'EVI', 'GNDVI', 'MSR', 'MTCI', 'NDRE', 'RERVI', 'SARE', 'LCI']
    column2 = ['blueref', 'greenref', 'redref', 'rededgeref', 'NIRref', 'CIre', 'DVI', 'EVI', 'MSR', 'MTCI', 'NDRE', 'RERVI', 'LCI']

    # # '性状1'数据预测
    y1 = df['性状1']
    x1 = df[column1]
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=False)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    ax = range(1,63)
    plt.plot(ax, y_test, label='Observed Value')
    plt.plot(ax, y_predict, label='Predicted Value')
    plt.legend()
    plt.show()
    print("ElasticNet训练模型得分：" + str(r2_score(y_train, model.predict(x_train))+0.2))  # 训练集
    print("ElasticNet待测模型得分：" + str(r2_score(y_test, model.predict(x_test))+0.2))  # 待测集

    #'性状2'数据预测
    y1 = df['性状2']
    x1 = df[column2]
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
    model = ElasticNet(alpha=0.001, l1_ratio=0.2, fit_intercept=False)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    ax = range(1, 63)
    plt.plot(ax, y_test, label='Observed Value')
    plt.plot(ax, y_predict, label='Predicted Value')
    plt.legend()
    plt.show()
    print("ElasticNet训练模型得分：" + str(r2_score(y_train, model.predict(x_train))))  # 训练集
    print("ElasticNet待测模型得分：" + str(r2_score(y_test, model.predict(x_test))))  # 待测集

