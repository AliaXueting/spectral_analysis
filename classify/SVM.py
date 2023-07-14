import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("classify_data.csv")
    print(df.shape)
    x_feature = df.iloc[:, 1:34]
    y_label = df.iloc[:, 0]


    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_label, test_size=0.2)
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)
    print("Accuracy is:", svm_model.score(x_test, y_test))

    # Plot the ROC curve
    y_pred = svm_model.predict(x_test)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.legend(loc=4)
    plt.show()

