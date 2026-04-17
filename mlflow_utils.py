import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_db_url():
    """
    Fetch the database URL from environment variables and format it for SQLAlchemy.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url

def set_alias_for_best_model(experiment_name='Diabetes_Cloud_Project',
                             alias_name='champion',
                             model_name='Diabetes_model'):
    """
    Identify the best performing model in an experiment and assign it a registered alias.

    Args:
        experiment_name (str): The name of the experiment to search within.
        alias_name (str): The alias to assign (e.g., 'champion', 'production').
        model_name (str): The name under which the model is registered in the Model Registry.
    """
    
    # 1. Setup connection to the remote Tracking Server / Database
    db_url = get_db_url()
    mlflow.set_tracking_uri(db_url)
    mlflow.set_registry_uri(db_url)
    
    client = MlflowClient()
    
    # 2. Retrieve experiment metadata
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # 3. Search for the run with the lowest MSE
    # Note: search_runs returns a pandas DataFrame by default
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_mse ASC"],
        max_results=1
    )

    if runs.empty:
        print("No runs found for this experiment.")
        return

    best_run_id = runs.iloc[0]['run_id']
    best_mse = runs.iloc[0]['metrics.test_mse']
    
    print(f"Found Best Run ID: {best_run_id} with MSE: {best_mse:.4f}")

    # 4. Register the model version from the best run
    model_uri = f'runs:/{best_run_id}/model'
    model_details = mlflow.register_model(model_uri, model_name)

    # 5. Assign the alias (e.g., @champion) to this specific version
    client.set_registered_model_alias(
        name=model_name, 
        alias=alias_name, 
        version=str(model_details.version)
    )

    print(f"Successfully registered model: {model_name} (v{model_details.version})")
    print(f"Alias '@{alias_name}' points to the best model now!")

if __name__ == '__main__':
    # Execution entry point
    set_alias_for_best_model()