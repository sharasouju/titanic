import pandas as pd
from sklearn.linear_model import LogisticRegression
import csv as csv

#テストデータの読み込み
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#不要なカラムの削除
train_df = train_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
test_df = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

#Ageの欠損値を男女それぞれの中央値で補完
age_train_mean = train_df.groupby("Sex").Age.mean()
age_test_mean = test_df.groupby("Sex").Age.mean()

def fage_train(x):
    if x.Sex == "male":
        return round(age_train_mean["male"])
    if x.Sex == "female":
        return round(age_train_mean["female"])

def fage_test(x):
    if x.Sex == "male":
        return round(age_test_mean["male"])
    if x.Sex == "female":
        return round(age_test_mean["female"])

#年齢の欠損値を男女それぞれの平均で補完
train_df.Age.fillna(train_df[train_df.Age.isnull()].apply(fage_train, axis=1), inplace=True)
test_df.Age.fillna(test_df[test_df.Age.isnull()].apply(fage_test, axis=1), inplace=True)

#女性が1となるようにダミー変数を設定
train_df["female"] = train_df["Sex"].map({"male":0, "female":1}).astype(int)
test_df["female"] = test_df["Sex"].map({"male":0, "female":1}).astype(int)

#Pclassをダミー変数で分ける
pclass_train_df = pd.get_dummies(train_df["Pclass"], prefix="Class")
pclass_test_df = pd.get_dummies(test_df["Pclass"], prefix="Class")

#Class_3を削除-->これも入れたほうが精度上がる
#pclass_train_df = pclass_train_df.drop(["Class_3"], axis=1)
#pclass_test_df = pclass_test_df.drop(["Class_3"], axis=1)

#Class_1、Class_2カラムを追加
train_df = train_df.join(pclass_train_df)
test_df = test_df.join(pclass_test_df)

#モデル生成
x = train_df.drop(["PassengerId", "Pclass", "Sex", "Survived"], axis=1)
y = train_df.Survived

clf = LogisticRegression()

#学習
clf.fit(x, y)

#スコア
print(clf.score(x, y))

#予測の実行
ids = test_df["PassengerId"].values
test_dt = test_df.drop(["PassengerId", "Pclass", "Sex"], axis=1)
test_predict = clf.predict(test_dt)

print(test_predict)

#データの書き出し(
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, test_predict))
submit_file.close
