import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm


def preprocessing(df):
    df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    return df


def main():
    train_df = pd.read_csv('../../data/titanic/train.csv')
    test_df = pd.read_csv('../../data/titanic/test.csv')
    test_df_copy = test_df.copy()

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_X = train_df.drop('Survived', axis=1)
    train_y = train_df['Survived']
    test_X = test_df.copy()

    # 前処理
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    model = svm.SVC()
    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)

    submission = pd.DataFrame({
        'PassengerId': test_df_copy['PassengerId'],
        'Survived': pred_y
    })

    submission.to_csv('submisson.csv', index=False)


if __name__ == '__main__':
    main()
