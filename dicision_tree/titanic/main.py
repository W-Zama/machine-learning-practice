import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV


def preprocessing(df):
    df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    return df


def main():
    df_train = pd.read_csv('../../data/titanic/train.csv')
    df_test = pd.read_csv('../../data/titanic/test.csv')
    df_test_copy = df_test.copy()

    df_train = preprocessing(df_train)
    df_test = preprocessing(df_test)

    X_train = df_train.drop('Survived', axis=1)
    y_train = df_train['Survived']
    X_test = df_test.copy()

    # clf = DecisionTreeClassifier(random_state=42)

    # ハイパーパラメータの設定
    max_depth = np.arange(1, 50)
    max_depth = np.concatenate([[None], max_depth])
    min_samples_split = np.arange(1, 50, 2)
    min_samples_leaf = np.arange(1, 50, 2)

    param_grid = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': ['gini', 'entropy']
    }

    # GridSearchCVの設定
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(
        random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy: {grid_search.best_score_:.2f}')

    # 最適なモデルの取得
    best_clf = grid_search.best_estimator_

    # テストデータで予測
    y_pred = best_clf.predict(X_test)

    # clf.fit(X_train, y_train)

    # # 決定木の可視化
    # plt.figure()
    # plot_tree(clf)
    # plt.show()

    # trainデータに対する評価
    # y_pred = clf.predict(X_train)
    # print(classification_report(y_train, y_pred))

    # actual prediciton
    # y_pred = clf.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': df_test_copy['PassengerId'],
        'Survived': y_pred
    })

    submission.to_csv('submission_with_grid_search.csv', index=False)


if __name__ == '__main__':
    main()
