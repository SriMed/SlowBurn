import pandas as pd
import json

df = pd.read_csv("train.csv")
df.fillna(df.mean(), inplace=True)
df = df.sample(n=200)

data_dict = {}



fatigue = list(df["Mental Fatigue Score"])
data_dict["Mental_Fatigue_Score"] = fatigue

Designation = list(df["Designation"])
data_dict["Designation"] = Designation

resource = list(df["Resource Allocation"])
data_dict["Resource_Allocation"] = resource

burn = list(df["Burn Rate"])
data_dict["Burn_Rate"] = burn

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)