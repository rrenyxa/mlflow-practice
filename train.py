import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv


load_dotenv()

def get_mlflow_configs():
    """Help function to get properly formatted db_url and backet name"""
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    return db_url, s3_bucket

def save_feature_importance(model, feature_names):
    importance = np.abs(model.coef_)
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_df = feat_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    plot_path = "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_with_extras(epochs=100, lr=0.01):
    
    db_url, s3_bucket = get_mlflow_configs()
    mlflow.set_tracking_uri(db_url)
    
    experiment_name = "Diabetes_Cloud_Project"
    artifact_location = f"s3://{s3_bucket}/mlflow-artifacts"

    
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, 
                                 artifact_location=artifact_location)
    
    mlflow.set_experiment(experiment_name)
    
    
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
        
        mlflow.log_text(results_df.to_csv(index=False), "results/test_predictions.csv")

        
        plot_file = save_feature_importance(model, data.feature_names)
        mlflow.log_artifact(plot_file, artifact_path="plots")
        os.remove(plot_file)

        
        mlflow.log_metric("test_mse", mean_squared_error(y_test, test_preds))
        mlflow.log_param("lr", lr)

        
        signature = infer_signature(X_test, test_preds)
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            signature=signature
        )
        
        print(f"Run finished for LR={lr}. Artifacts uploaded to S3!")

if __name__ == "__main__":
    learning_rates = [0.1, 0.01]
    for lr in learning_rates:
        train_with_extras(lr=lr)