

---

# NYC Vehicle Clustering and Demand Prediction

This project analyzes GPS data of NYC vehicles using clustering and regression techniques. It performs data preprocessing, spatial clustering, and temporal demand prediction to understand vehicle distribution and trends.

## Features

* **Data Preprocessing**: Combines and parses multi-month GPS datasets into structured time-based features.
* **Clustering**: Applies KMeans to identify spatial clusters based on vehicle GPS coordinates.
* **Regression Analysis**: Trains models (LightGBM, XGBoost, Random Forest) to predict hourly vehicle demand per cluster.
* **Visualization**: Generates cluster maps and bar charts showing hourly demand.
* **MLflow Integration**: Tracks experiments, logs models, metrics, and artifacts.
* **Reporting**: Outputs performance scores and visual reports.

## Requirements

* Python 3.8+
* Libraries:
  `polars`, `mlflow`, `joblib`, `lightgbm`, `xgboost`, `folium`, `scikit-learn`, `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── code.py                   # Main pipeline script
├── data/
│   ├── raw/                  # Raw input CSVs (April–September)
│   ├── processed/            # Preprocessed data
│   └── clusters/             # Clustered data
├── models/                   # Saved clustering and regression models
├── reports/
│   ├── figures/              # Cluster maps and demand plots
│   └── tests/                # Clustering and regression test metrics
└── README.md                 # Project documentation
```

## How to Run

1. Ensure `mlflow` is running:

   ```bash
   mlflow ui
   ```

2. Execute the script:

   ```bash
   python code.py
   ```

> Note: Uncomment relevant sections in `main()` to reprocess data or retrain models.

## Outputs

* **Clustering Scores**: Calinski-Harabasz and Davies-Bouldin indices.
* **Regression Metrics**: RMSE for each model and cluster.
* **Maps & Plots**: Cluster visualization and hourly demand bar charts.


---

