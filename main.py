import pandas as pd
import polars as pl
import joblib
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklego.preprocessing import RepeatingBasisFunction

pl.read_csv("data/processed/data.csv")[:100].write_csv("data/processed/reduced.csv")


def predict(data: pd.DataFrame, neighbor_no: int = 10):
    kn = KMeans(n_clusters=neighbor_no)
    kn.fit(data)
    return kn


def process(data: pd.DataFrame, name: str):
    data["Date/Time"] = pd.to_datetime(data["Date/Time"])
    data["Hour"] = data["Date/Time"].dt.hour
    data["Day"] = data["Date/Time"].dt.day
    data["Year"] = data["Date/Time"].dt.year
    data["Month"] = data["Date/Time"].dt.month
    data["Minute"] = data["Date/Time"].dt.minute
    data.to_csv("data/processed/data_intermediate.csv")
    hour = RepeatingBasisFunction(column="Hour", input_range=(0, 23))
    hour.fit(data)
    joblib.dump(hour, "models/scalers/hour_rbf.joblib")
    hour.transform(data)
    minute = RepeatingBasisFunction(column="Minute", input_range=(0, 60))
    minute.fit(data)
    joblib.dump(minute, "models/scalers/hour_rbf.joblib")
    minute.transform(data)
    day = RepeatingBasisFunction(column="Day", input_range=(0, 365))
    day.fit(data)
    joblib.dump(minute, "models/scalers/minute_rbf.joblib")
    day.transform(data)
    month = RepeatingBasisFunction(column="Month", input_range=(0, 12))
    month.fit(data)
    joblib.dump(month, "models/scalers/month_rbf.joblib")
    month.transform(data)
    le = LabelEncoder()
    data["Base"] = le.fit_transform(data["Base"])
    data = data.drop(["Date/Time"], axis=1)
    for i in data.columns:
        scaler = StandardScaler()
        scaler.fit(data[[i]])
        joblib.dump(scaler, filename=f"models/scalers/{i}.joblib")
        data[i] = scaler.transform(data[[i]])
    data.to_csv(f"data/processed/{name}.csv")


# """
april = pd.read_csv("data/raw/data-apr14.csv")
may = pd.read_csv("data/raw/data-may14.csv")
june = pd.read_csv("data/raw/data-jun14.csv")
july = pd.read_csv("data/raw/data-jul14.csv")
august = pd.read_csv("data/raw/data-aug14.csv")
september = pd.read_csv("data/raw/data-sep14.csv")
months = {
    "april": april,
    "may": may,
    "june": june,
    "july": july,
    "august": august,
    "september": september,
}
total_data = pd.DataFrame()
for month, data in months.items():
    total_data = pd.concat([total_data, data])
process(total_data, "data")
# """


def dump_models():
    data = pd.read_csv("data/processed/data.csv")
    model = predict(data)
    joblib.dump(model, filename=f"models/clustering.joblib")


def dump_clusters():
    model = joblib.load(f"models/clustering.joblib")
    data = pd.read_csv(f"data/processed/data.csv")
    cluster = pd.concat([data, pd.Series(model.predict(data), name="cluster")], axis=1)
    cluster.to_csv("data/clusters/clustered_data.csv")
    print(cluster)
    cluster_frames = {}
    for c in cluster["cluster"].unique():
        cluster_frames[f"cluster {c}"] = cluster[cluster["cluster"] == c]
        cluster_frames[f"cluster {c}"].to_csv(f"data/clusters/cluster {c}.csv")


def regression(
    data,
    target=["cluster"],
):
    data = pd.read_csv("data/clusters/clustered_data.csv")
    lr = LinearRegression()
    lr.fit(data, data[target])
    joblib.dump(lr, "models/regression.joblib")


# regression(pd.read_csv("data/clusters/clustered_data.csv"))
# dump_models()
# dump_clusters()
