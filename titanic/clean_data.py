import pandas as pd

def clean(df: pd.DataFrame):
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df = df.drop(columns=["Cabin"])
    
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    common_list = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master"
    }

    df["Title"] = df["Title"].apply(lambda x: common_list.get(x, "Rare"))

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = df["FamilySize"].apply(lambda x: 1 if x == 1 else 0) # alternatively, df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 80], labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

    df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}) # alternative: df["Sex"].apply(lambda x: 0 if x == "male" else 1)

    df = pd.get_dummies(df, columns=["Embarked", "Title", "AgeGroup", "FareBin"], drop_first=True, dtype=int)

    drop_cols = ["PassengerId", "Name", "Ticket"]

    df = df.drop(columns=drop_cols)

    return df