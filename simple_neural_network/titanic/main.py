import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# データの読み込み
train_df = pd.read_csv('../../data/titanic/train.csv')
test_df = pd.read_csv('../../data/titanic/test.csv')
test_df_copy = test_df.copy()

# 特徴量の作成
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

# 前処理パイプラインの定義
def preprocessing(df):
    df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

    numerical_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'IsAlone']

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor.fit_transform(df)


# データの前処理
train_X = preprocessing(train_df)
test_X = preprocessing(test_df)
train_y = train_df['Survived']

# モデルの定義
model = Sequential()
model.add(Dense(64, input_dim=train_X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# モデルのコンパイルと学習
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=500, batch_size=32, validation_split=0.2)

pred_y = model.predict(test_X).reshape(-1)
pred_y_as_binary = (pred_y > 0.5).astype("int32")

submission = pd.DataFrame({
    'PassengerId': test_df_copy['PassengerId'],
    'Survived': pred_y_as_binary
})

submission.to_csv('submisson_better.csv', index=False)