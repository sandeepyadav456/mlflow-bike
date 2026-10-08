import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

data = pd.DataFrame(
    {
        "temp": np.random.rand(500),
        "humidity": np.random.rand(500),
        "windspeed": np.random.rand(500),
        "hour": np.random.randint(0, 24, 500),
        "count": np.random.randint(50, 500, 500),
    }
)

X = data.drop("count", axis=1)
y = data["count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    ),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bike-demand-nested-model-comparison")
mlflow.autolog()

with mlflow.start_run(run_name="baseline_models"):

    mlflow.set_tag("project", "bike_sharing_demand")
    mlflow.set_tag("experiment_type", "model_comparison")

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name, nested=True):

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_param("algorithm", model_name)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="pickle")
            print(f"{model_name} RMSE: {rmse:.2f}")