import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# 1. Load and split the dataset
data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# 2. Create an MLflow experiment
mlflow.set_experiment("California Housing Experiments")


# 3. Evaluate the trained model
def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)

    return {
        "RMSE": root_mean_squared_error(y_val, predictions),
        "MAE": mean_absolute_error(y_val, predictions),
        "R2": r2_score(y_val, predictions),
    }


# 4. Train and track one experiment
def execute(max_depth, learning_rate):
    run_name = f"depth_{max_depth}_lr_{learning_rate}"

    with mlflow.start_run(run_name=run_name):

        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val)

        # Log hyperparameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"\nCompleted: {run_name}")
        print("Metrics:", metrics)


# 5. Run the three experiments
execute(max_depth=3, learning_rate=0.1)
execute(max_depth=5, learning_rate=0.05)
execute(max_depth=7, learning_rate=0.01)

print("\nAll experiments completed!")