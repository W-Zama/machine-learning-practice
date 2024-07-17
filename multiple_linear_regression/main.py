import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


def main():
    df = pd.read_csv('../data/Student_Performance.csv')

    # 前処理
    encoder = OneHotEncoder(sparse_output=False)
    encoded_activities = encoder.fit_transform(
        df[['Extracurricular Activities']])
    encoded_df = pd.DataFrame(encoded_activities, columns=encoder.get_feature_names_out(
        ['Extracurricular Activities']))
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop('Extracurricular Activities', axis=1)

    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # 結果の表示
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='blue', linewidth=3, label='Ideal fit')
    plt.xlabel('Actual Performance Index')
    plt.ylabel('Predicted Performance Index')
    plt.title('Actual vs Predicted Performance Index')
    plt.legend()
    plt.savefig(
        '../multiple_linear_regression/figure/student_performance_prediction.png')


if __name__ == '__main__':
    main()
