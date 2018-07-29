import pandas as pd
import csv as csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv("train.csv", header=0)

# Convert "Sex" to be a dummy variable (female = 0, Male = 1)
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head(3)

# "Age"の欠損値を、"Age"の中央値で補完
median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
  train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# remove un-used columns
train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
train_df.head(3)

# Load test data, Convert "Sex" to be a dummy variable
test_df = pd.read_csv("test.csv", header=0)
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Complement the missing values of "Age" column with average of "Age"
median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
  test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# Copy test data's "PassengerId" column, and remove un-used columns
ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
test_df.head(3)

# Predict with "Random Forest"
train_data = train_df.values
test_data = test_df.values

#パラメータの最適化
parameters = {
        'n_estimators':[10, 50, 100, 110, 120, 130, 140, 150, 160, 170],
        'max_features':[5]
}

#print("パラメータ推定開始")

#model = GridSearchCV(RandomForestClassifier(), parameters)
model = RandomForestClassifier(n_estimators=150)
output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

#最適化したパラメータの表示
#print("\nパラメータ\n")
#print(model.best_params_)



#特徴量の重要度
feature = model.feature_importances_

#重要度を上から順に出力する。
f = pd.DataFrame({'number': range(0, len(feature)), 'feature': feature[:]})

#特徴量の名前
label = train_df.columns[1:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, output))
submit_file.close()
