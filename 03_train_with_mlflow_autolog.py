import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow

df = pd.read_csv("data/train.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

target = "count"
drop_cols = ["datetime", "count", "casual", "registered"]
X = df.drop(columns=drop_cols)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1
    }

mlflow.set_experiment("bike-demand-tracking")

mlflow.autolog()

with mlflow.start_run(run_name="rf_baseline"):
    trainedmodel = RandomForestRegressor(**params)
    trainedmodel.fit(X_train, y_train)