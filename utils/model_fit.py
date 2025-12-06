import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def fit_knn_regressor(
    X_train,
    y_train,
    n_neighbors: int = 5,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    knn.fit(X_train_scaled, y_train)

    return knn, scaler