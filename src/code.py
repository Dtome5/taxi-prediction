import polars as pl
import pandas as pd
import joblib
import numpy as np
import folium
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklego.preprocessing import RepeatingBasisFunction
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    mean_squared_error,
    calinski_harabasz_score,
    r2_score,
    davies_bouldin_score,
)

# pl.read_csv("data/raw/data-apr14.csv")[:100].write_csv("data/processed/reduced.csv")


def process(data: pl.DataFrame, name: str):
    data = data.with_columns(
        pl.col("Date/Time").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
    )
    data = data.with_columns(pl.col("Date/Time").dt.hour().alias("Hour"))
    data = data.with_columns(pl.col("Date/Time").dt.day().alias("Day"))
    data = data.with_columns(pl.col("Date/Time").dt.month().alias("Month"))
    data = data.with_columns(pl.col("Date/Time").dt.year().alias("Year"))
    data = data.with_columns(pl.col("Date/Time").dt.minute().alias("Minute"))
    data.write_csv("data/processed/data_intermediate.csv")
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
    data = data.with_columns(
        pl.Series(name="Base", values=le.fit_transform(data["Base"]))
    )
    data = data.drop(["Date/Time"])
    # for i in data.columns:
    scaler = StandardScaler()
    scaler.fit(data)
    joblib.dump(scaler, filename="models/scalers/scaler.joblib")
    # data = data.with_columns(pl.lit(scaler.transform(data).flatten()).alias(i))
    print(data, data.dtypes)
    joblib.dump(data, f"data/processed/{name}.joblib")
    data.write_csv(f"data/processed/{name}.csv")


"""
april = pl.read_csv("data/raw/data-apr14.csv")
may = pl.read_csv("data/raw/data-may14.csv")
june = pl.read_csv("data/raw/data-jun14.csv")
july = pl.read_csv("data/raw/data-jul14.csv")
august = pl.read_csv("data/raw/data-aug14.csv")
september = pl.read_csv("data/raw/data-sep14.csv")
months = {
    "april": april,
    "may": may,
    "june": june,
    "july": july,
    "august": august,
    "september": september,
}
total_data = pl.DataFrame()
for month, data in months.items():
    total_data = pl.concat([total_data, data])
print(total_data)
process(total_data, "data")
# """


def predict(data: pl.DataFrame, neighbor_no: int = 10):
    kn = KMeans(n_clusters=neighbor_no, random_state=0)
    kn.fit(data)
    nbrs = NearestNeighbors(n_neighbors=10)
    nbrs.fit(data)
    return kn


def dump_models():
    data = pl.read_csv("data/processed/data.csv")
    model = predict(data)
    joblib.dump(model, filename="models/clustering.joblib")


def dump_clusters():
    knc = KNeighborsClassifier()

    # Load data and clustering model
    data = pl.read_csv("data/processed/data.csv")
    model = joblib.load("models/clustering.joblib")

    # Predict clusters using the loaded model and concatenate with original data
    clusters = pl.DataFrame({"cluster": model.predict(data)})
    cluster = pl.concat([data, clusters], how="horizontal")

    print("training knc")
    # Fit KNeighborsClassifier: Apply ravel() to the target variable
    # to ensure it is a 1D array as expected by scikit-learn
    knc.fit(data, model.predict(data).ravel())
    joblib.dump(knc, "models/knc_clusters.joblib")
    print("end knc training")

    # Predict clusters using the trained KNC model
    knc_clusters = pl.DataFrame({"cluster": knc.predict(data)})
    knc_data = pl.concat([data, knc_clusters], how="horizontal")

    # Write KNC clustered data to CSV
    knc_data.write_csv("data/clusters/clustered_data.csv")

    # Load intermediate data and concatenate with KNC clusters, then write to CSV
    data_int = pl.read_csv("data/processed/data_intermediate.csv")
    pl.concat([data_int, knc_clusters], how="horizontal").write_csv(
        "data/clusters/clustered_alt.csv"
    )

    # Write original clustering model data to CSV
    cluster.write_csv("data/clusters/kmeans_data.csv")

    # Print unique clusters from the original clustering model
    print(
        cluster["cluster"].unique(),
    )

    # Separate data by cluster from the original clustering model and write each to CSV
    cluster_frames = {}
    for c in cluster["cluster"].unique():
        cluster_frames[f"cluster {c}"] = cluster.filter(pl.col("cluster") == c)
        cluster_frames[f"cluster {c}"].write_csv(f"data/clusters/cluster {c}.csv")

    # Calculate clustering scores using the KNC results
    chscore = calinski_harabasz_score(data, knc.predict(data).ravel())
    dbscore = davies_bouldin_score(data, knc.predict(data).ravel())

    # Write scores to text files
    file_chscore = open("tests/chscore.txt", "w+")
    file_dbscore = open("tests/dbscore.txt", "w+")
    file_chscore.write(f"{chscore}")
    file_dbscore.write(f"{dbscore}")
    file_chscore.close()
    file_dbscore.close()


def map():
    model = joblib.load("models/clustering.joblib")
    latscaler = joblib.load("models/scalers/Lat.joblib")
    lonscaler = joblib.load("models/scalers/Lon.joblib")
    scaled_cluster_centers = model.cluster_centers_
    lats = latscaler.inverse_transform(scaled_cluster_centers[:, [0]])
    lons = lonscaler.inverse_transform(scaled_cluster_centers[:, [1]])
    # cluster_centers = [lats, lons]
    mean_cluster_centers = [
        float(lats.mean()),
        float(lons.mean()),
    ]
    mymap = folium.Map(location=mean_cluster_centers)
    folium.Marker(
        location=mean_cluster_centers,
        popup=f"Mean Location\nLat: {mean_cluster_centers[0]:.4f}, Lon: {mean_cluster_centers[1]:.4f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(mymap)

    cluster_colors = [
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "beige",
        "darkgreen",
        "red",
        "brown",
        "black",
    ]

    for i in range(len(lats)):
        print([float(lats[i][0]), float(lons[i][0])])
        folium.CircleMarker(
            location=[float(lats[i][0]), float(lons[i][0])],
            radius=7,
            color=cluster_colors[i],
            tooltip=f"Cluster {i}",
            # fill=True,
            # fill_color="red",
            fill_opacity=0.8,
            popup=f"Cluster {i}\nLat: {float(lats[i][0]):.4f}, Lon: {float(lons[i][0]):.4f}",
        ).add_to(mymap)
    nyc_bounds = [
        [lats.max(), lons.max()],
        [lats.min(), lons.min()],
    ]  # [SW, NE] corners
    print(nyc_bounds)
    mymap.fit_bounds(nyc_bounds)
    mymap.save("map.html")


def regression(
    data,
    target=["cluster"],
):
    data = pl.read_csv("data/clusters/clustered_data.csv")
    print(data.group_by("cluster").len())
    data = data.group_by(["Hour", "cluster"]).len()
    models = {}
    mse_errors = {}
    r2 = {}
    for cluster in data["cluster"].unique():
        data_cluster = data.filter(pl.col("cluster") == cluster).drop("cluster")
        X = data_cluster[["Hour"]]
        y = data_cluster["len"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, "models/regression.joblib")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        models[cluster] = {"model": model, "mse": mse}
        mse_errors[f"Cluster {cluster}"] = mse
        r2_value = r2_score(y_test, y_pred)
        r2[f"Cluster {cluster}"] = r2_value
    for cluster, result in models.items():
        print(f"Cluster {cluster}: MSE = {result['mse']:.2f}")
    file_mse = open("tests/regression/Mse.json", "w+")
    file_acc = open("tests/regression/acc_score.json", "w+")
    json.dump(r2, file_acc)
    json.dump(mse_errors, file_mse)
    file_mse.close()


def plots():
    data = pl.read_csv("data/clusters/clustered_alt.csv")
    # data[:100].write_csv("data/clusters/redcalt.csv")
    data_agg = data.group_by(["cluster", "Hour"]).len()
    for i in data_agg["cluster"].unique():
        filtered = data_agg.filter(pl.col("cluster") == i)
        fig, ax = plt.subplots()
        ax.bar(filtered["Hour"], filtered["len"])
        ax.set_title(f"Cluster {i}")
        fig.savefig(f"reports/figures/cluster {i}")


# plots()
dump_models()
dump_clusters()
regression(pd.read_csv("data/clusters/clustered_data.csv"))
# map()
