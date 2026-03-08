import pandas as pd

df = pd.read_csv("video_games_sales.csv")

#print(df.info)
print("NUll values---------------")
print(df.isnull().sum()) #missing values check!

print("\nHead ---------------")
print(df.head()) #First 5 row

df = df.dropna(subset=["year"])
print(df.info)

df = df.dropna(subset=["year"])

df["publisher"] = df["publisher"].fillna("unknwon")

df["year"] = df["year"].astype(int)

df["hit"] = df["global_sales"].apply(lambda x:1 if x>1.0 else 0)
print(df["hit"].value_counts())

genre_hit = df.groupby("genre")["hit"].mean().sort_values(ascending=False)
print(genre_hit)

platform_hit = df.groupby("platform")["hit"].mean().sort_values(ascending=False)
print(platform_hit)

features = ["platform","genre","publisher","year"]
x = df[features]
y = df["hit"]

X = pd.get_dummies(x,columns=["platform","genre","publisher"],drop_first=True) #one-hot encoding
print("X:",X)