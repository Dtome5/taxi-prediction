import json
import mlflow
import folium
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import polars as pl
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# function to prepare data for machine learning
def process(data: pl.DataFrame, name: str = "data"):
    with mlflow.start_run(run_name="process_data"):
        data = data.with_columns(
            pl.col("Date/Time").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
        )
        # extract features from Date/Time
        data = data.with_columns(pl.col("Date/Time").dt.hour().alias("Hour"))
        data = data.with_columns(pl.col("Date/Time").dt.day().alias("Day"))
        data = data.with_columns(pl.col("Date/Time").dt.month().alias("Month"))
        data = data.with_columns(pl.col("Date/Time").dt.year().alias("Year"))
        data = data.with_columns(pl.col("Date/Time").dt.minute().alias("Minute"))
        data = data.drop(["Date/Time"])
        # save data
        joblib.dump(data, f"data/processed/{name}.joblib")
        data.write_csv(f"data/processed/{name}.csv")
        # Log artifact
        mlflow.log_artifact(f"data/processed/{name}.csv", "processed data")


def train_clusters(
    data: pl.DataFrame = pl.read_csv("data/processed/data.csv"), neighbor_no: int = 10
):
    with mlflow.start_run(run_name="Clustering"):
        km = KMeans(n_clusters=neighbor_no, random_state=0)
        coordinates = data["Lat", "Lon"]
        km.fit(coordinates)
        joblib.dump(km, filename="models/km_clustering.joblib")
        # Calculate clustering scores using the KNC results
        chscore = calinski_harabasz_score(coordinates, km.labels_)
        dbscore = davies_bouldin_score(coordinates, km.labels_)
        # Write scores to text files
        file_chscore = open("reports/tests/clustering/chscore.txt", "w+")
        file_dbscore = open("reports/tests/clustering/dbscore.txt", "w+")
        file_chscore.write(f"{chscore}")
        file_dbscore.write(f"{dbscore}")
        file_chscore.close()
        file_dbscore.close()
        # Log parameters and metrics
        mlflow.log_param("n_clusters", neighbor_no)
        mlflow.log_metric("calinski_harabasz", chscore)
        mlflow.log_metric("davies_bouldin", dbscore)
        # Log model
        mlflow.sklearn.log_model(km, "clustering_model")
        mlflow.end_run()
    print("done training model")


def dump_clusters():
    with mlflow.start_run(run_name="generate_clusters"):
        # Load data and clustering model
        data = pl.read_csv("data/processed/data.csv")
        model = joblib.load("models/km_clustering.joblib")

        # Predict clusters using the loaded model and concatenate with original data
        cluster = pl.DataFrame({"cluster": model.predict(data["Lat", "Lon"])})
        clusters = pl.concat([data, cluster], how="horizontal")
        clusters.write_csv("data/clusters/kmeans_data.csv")

        # Print unique clusters from the original clustering model
        print(
            cluster["cluster"].unique(),
        )

        # Separate data by cluster from the original clustering model and write each to CSV
        cluster_frames = {}
        for c in clusters["cluster"].unique():
            cluster_frames[f"cluster {c}"] = clusters.filter(pl.col("cluster") == c)
            cluster_frames[f"cluster {c}"].write_csv(f"data/clusters/cluster {c}.csv")
        mlflow.log_artifact("data/clusters/kmeans_data.csv", "cluster_dataframes")
        mlflow.end_run()


def draw_map():
    model = joblib.load("models/km_clustering.joblib")
    cluster_centers = model.cluster_centers_
    lats = cluster_centers[:, [0]]
    lons = cluster_centers[:, [1]]
    print(cluster_centers)
    mean_cluster_centers = [
        float(lats.mean()),
        float(lons.mean()),
    ]

    # draw map with folium
    mymap = folium.Map(location=mean_cluster_centers)
    folium.Marker(
        location=mean_cluster_centers,
        popup=f"Mean Location\nLat: {mean_cluster_centers[0]:.4f}, Lon: {mean_cluster_centers[1]:.4f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(mymap)

    for i in range(len(lats)):
        folium.CircleMarker(
            location=[float(lats[i][0]), float(lons[i][0])],
            radius=17,
            color="purple",
            tooltip=f"Cluster {i}",
            fill=True,
            fill_color="purple",
            fill_opacity=0.8,
            popup=f"Cluster {i}\nLat: {float(lats[i][0]):.4f}, Lon: {float(lons[i][0]):.4f}",
        ).add_to(mymap)
    # define boundaries
    nyc_bounds = [
        [lats.max(), lons.max()],
        [lats.min(), lons.min()],
    ]
    mymap.fit_bounds(nyc_bounds)
    mymap.save("reports/figures/map.html")


# function to add rmse to dict
def rmse(true, pred, dict, key):
    rmse = root_mean_squared_error(true, pred)
    dict[key] = rmse


# saving test results
def save_test(dict, name):
    file = open(f"reports/tests/regression/{name}", "w+")
    json.dump(dict, file)
    file.close()


def lgboost_train(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="LightGBM"):
        # tune model
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            # Learning parameters
            "learning_rate": 0.03,
            # "n_estimators": 300,
            "num_iterations": 300,
            # Tree structure parameters
            "random_state": 0,
            "num_leaves": 70,
            "max_depth": 7,
            "max_bin": 24,
            "bagging_freq": 1,
            "bagging_fraction": 0.8,
            "min_child_samples": 5,
            # Feature sampling parameters
            "colsample_bytree": 1.0,  # Only 2 features, use all
            "subsample": 0.9,
            "subsample_freq": 1,
            # Regularization parameters
            "reg_alpha": 0.05,
            "reg_lambda": 0.1,
            # Target has extreme values, using robust loss
            "huber_delta": 5.0,
            # "extra_trees": False,
            # Categorical feature handling
            "categorical_feature": ["name: cluster Hour"],
            # Other parameters
            "verbose": -1,
        }
        gst = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train),
            valid_sets=lgb.Dataset(X_test, label=y_test),
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )
        # log model parameters and save model
        mlflow.log_params(params)
        mlflow.sklearn.log_model(gst, "XGBoost")
        joblib.dump(gst, "models/lgb_rergressor.joblib")
        mlflow.end_run()
    return gst


def randf_train(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="train_random_forest"):
        randf = RandomForestRegressor(random_state=0)
        randf.fit(X_train, y_train)
        joblib.dump(randf, "models/random_forest_regression.joblib")
        # log model parameters and save model
        mlflow.sklearn.log_model(randf, "RandomForestRegressor")
        mlflow.end_run()
    return randf


def xgb_train(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="train_xgboost"):
        xgbr = XGBRegressor()
        xgbr.fit(X_train, y_train)
        # log model parameters and save model
        mlflow.sklearn.log_model(xgbr, "XGBoostRegressor")
        joblib.dump(xgbr, "models/xgb_regression.joblib")
        mlflow.end_run()
    return xgbr


def regression(data=pl.read_csv("data/clusters/kmeans_data.csv")):
    data = pl.read_csv("data/clusters/kmeans_data.csv")
    data = data.group_by(["Hour", "cluster"]).len().sort("cluster")
    randf_rmse = {}
    lgb_rmse = {}
    xgbr_rmse = {}
    X = data[["Hour", "cluster"]]
    y = data["len"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    lgbr = lgboost_train(
        X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    )
    xgbr = xgb_train(X_train, y_train, X_test, y_test)
    randf = randf_train(X_train, y_train, X_test, y_test)
    mse = {
        "RandomForestRegressor": root_mean_squared_error(y_test, randf.predict(X_test)),
        "LightGBMRegressor": root_mean_squared_error(
            y_test, lgbr.predict(X_test.to_numpy())
        ),
        "XGBoostRegressor": root_mean_squared_error(y_test, xgbr.predict(X_test)),
    }
    xy = pl.concat([pl.DataFrame(X_test), pl.DataFrame(y_test)], how="horizontal")
    for cluster in xy["cluster"].unique():
        sample_x = xy.filter(pl.col("cluster") == cluster)[["Hour", "cluster"]]
        sample_y = xy.filter(pl.col("cluster") == cluster)[["len"]]
        print(sample_x)
        randf_pred = randf.predict(sample_x)
        xgbr_pred = xgbr.predict(sample_x)
        lgb_pred = lgbr.predict(sample_x.to_numpy())
        rmse(sample_y, randf_pred, randf_rmse, f"Cluster {cluster}")
        rmse(sample_y, xgbr_pred, xgbr_rmse, f"Cluster {cluster}")
        rmse(sample_y, lgb_pred, lgb_rmse, f"Cluster {cluster}")
    with mlflow.start_run(run_name="Regression"):
        mlflow.log_metrics(mse)
        mlflow.end_run()
    save_test(randf_rmse, "Mse_randf.json")
    save_test(xgbr_rmse, "Mse_xgbr.json")
    save_test(lgb_rmse, "Mse_lgb.json")


def plots():
    data = pl.read_csv("data/clusters/kmeans_data.csv")
    data_agg = data.group_by(["cluster", "Hour"]).len().sort("cluster")
    for i in data_agg["cluster"].unique():
        filtered = data_agg.filter(pl.col("cluster") == i)
        fig, ax = plt.subplots(dpi=300)
        ax.bar(filtered["Hour"], filtered["len"])
        ax.set_title(f"Cluster {i + 1}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Number of Vehicles")
        fig.savefig(f"reports/figures/cluster {i + 1}")


def main():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("nyc_vehicle_clusters")
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
    """
    # train_clusters()
    # dump_clusters()
    regression()
    draw_map()
    plots()
    data = pl.read_csv("data/clusters/kmeans_data.csv")
    print("mean: ", data.mean(), "max: ", data.max(), "min: ", data.min())
    monthly_demand = (
        data.group_by(["Month"]).len().sort(["Month"]).write_json("monthly_demand.json")
    )
    data.group_by("cluster").len().describe().write_json("data_describe.json")
    print(monthly_demand, data["Base"].n_unique())


main()
