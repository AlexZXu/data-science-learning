import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.read_csv("dataSP23.csv")
df["room_type"].value_counts()
df = pd.get_dummies(df, columns=["room_type"], dtype=float) #one-hot encode
 
df = df.drop(columns=["host_name", "host_id", "id", "name"])
df = pd.get_dummies(df, columns=["neighbourhood_group"], dtype=float)

enc = OrdinalEncoder()
df["neighbourhood"] = enc.fit_transform(df[["neighbourhood"]])

df.head(4)
df["last_review"] = pd.to_datetime(df["last_review"]).astype("int64")
df.info()