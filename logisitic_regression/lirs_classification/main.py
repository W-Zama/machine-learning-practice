import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


def main():
    df = pd.read_csv('../../data/iris.csv')

    # エンコード
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])

    # 入力と出力の整理
    X = df.drop(['Species', 'Id'], axis=1)
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, random_state=0)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'f1: {f1}')
    print(report)


if __name__ == '__main__':
    main()
