import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import mlflow
from mlflow.models.signature import infer_signature

def save_feature_importance(model, feature_names):
   #weights for SGD are in .cofes
    importance = np.abs(model.coef_)
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_df = feat_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_df)
    plt.title('Feature Importance (Absolute Coefficients)')
    plt.tight_layout()
    
  
    plot_path = "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_with_extras(epochs=100, lr=0.01):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes_Advanced_Logging")
    
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        model = SGDRegressor(eta0=lr, max_iter=epochs, random_state=42)
        model.fit(X_train_scaled, y_train)
        
    
        test_preds = model.predict(X_test_scaled)
        results_df = pd.DataFrame({
            'real_value': y_test,
            'prediction': test_preds,
            'error': y_test - test_preds
        })
        results_df.to_csv("test_predictions.csv", index=False)
        mlflow.log_artifact("test_predictions.csv", artifact_path="results")


        plot_file = save_feature_importance(model, data.feature_names)
        mlflow.log_artifact(plot_file, artifact_path="plots")

        # Метрики и модель
        mlflow.log_metric("test_mse", mean_squared_error(y_test, test_preds))
        signature = infer_signature(X_train, test_preds)
        mlflow.sklearn.log_model(sk_model=model, 
                                 artifact_path="model",
                                 signature=signature)
        
        print("Predictinos and feature importance graph are saved!")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    learning_rates = [0.5, 0.1, 0.01, 0.005]
    for lr in learning_rates:
        train_with_extras(lr=lr)