import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


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

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': df_test_copy['PassengerId'],
        'Survived': y_pred
    })

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
