import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay


def main():
    df = pd.read_csv('../../data/Iris.csv')

    # 前処理
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])

    X = df.drop(['Species', 'Id'], axis=1)
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, random_state=0)

    # 標準化
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    model = svm.SVC(C=10)
    model.fit(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)

    y_pred = model.predict(X_test_pca)

    # print(classification_report(y_test, y_pred))

    # 決定境界の描画
    DecisionBoundaryDisplay.from_estimator(model,
                                           X_test_pca,
                                           plot_method='contour',
                                        #    cmap=plt.cm.Paired,
                                        #    levels=[-1, 0, 1],
                                        #    alpha=0.5,
                                        #    xlabel='first principal component',
                                        #    ylabel='second principal component',
                                           )

    # # 学習データの描画
    for i, color in zip(model.classes_, 'bry'):
        idx = np.where(y_train == i)
        plt.scatter(
            X_train_pca[idx, 0],
            X_train_pca[idx, 1],
            c=color,
            edgecolor='black',
            s=20,
        )

    plt.show()


if __name__ == '__main__':
    main()
