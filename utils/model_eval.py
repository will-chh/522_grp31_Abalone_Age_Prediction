import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_knn(
    knn,
    scaler,
    X_train,
    y_train,
    X_test,
    y_test,
    plot: bool = True,
):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("=== KNN Regression performance (k=5) ===")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test  RMSE : {test_rmse:.4f}")

    if plot:
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_test_pred, alpha=0.5)

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--",
            color="black",
            label="Perfect Prediction Line",
        )

        plt.xlabel("Actual number of rings (True age proxy)")
        plt.ylabel("Predicted number of rings")
        plt.title("KNN model performance: actual vs predicted rings")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return train_rmse, test_rmse