import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('../data/SAT_and_GPA.csv')

df = pd.DataFrame(data)

X = df[['SAT']]
y = df['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'mse: {mse:.4f}')
print(f'r2: {r2:.4f}')

# 学習のプロット
plt.figure()
plt.scatter(X_train.values, y_train.values, label='actual', color='black')
plt.plot(X_train.values, model.predict(X_train), label='predicted', color='blue')
plt.xlabel('STA Score')
plt.ylabel('GPA')
plt.legend()
plt.savefig('figure/training.png')

# 予測のプロット
plt.figure()
plt.scatter(X_test.values, y_test.values, label='actural', color='black')
plt.plot(X_test.values, y_pred, label='predicted', color='blue')
plt.xlabel('STA Score')
plt.ylabel('GPA')
plt.legend()
plt.savefig('figure/prediction.png')
