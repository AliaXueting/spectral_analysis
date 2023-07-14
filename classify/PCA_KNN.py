import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("classify_data.csv")
    print(df.shape)
    x_feature = df.iloc[:, 1:34]
    y_label = df.iloc[:, 0]
    pca = PCA(n_components=30)
    new_feature = pca.fit_transform(x_feature)


    x_train, x_test, y_train, y_test = train_test_split(new_feature, y_label, test_size=0.2)
    knnModel = KNeighborsClassifier()
    knnModel.fit(x_train, y_train)
    print("Accuracy is:", knnModel.score(x_test, y_test))

    # Plot the ROC curve
    y_pred = knnModel.predict(x_test)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.legend(loc=4)
    plt.show()

