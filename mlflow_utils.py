import mlflow
from mlflow.tracking import MlflowClient

def set_alis_for_best_model(uri='sqlite:///mlflow.db',
                            experiment_name='Diabetes_Advanced_Logging',
                            alias_name='champion'):
    
    mlflow.set_registry_uri(uri=uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.`test_mse` ASC"],
        max_results=1
    )
    best_run_id  = runs['run_id'][0]
    best_mse = runs['metrics.test_mse'][0]
    print(best_run_id, best_mse)
    model_name = 'Diabetes_model'
    model_uri = f'runs:/{best_run_id}/model'
    model_details = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(name=model_name, 
                                  alias=alias_name, 
                                  version=str(model_details.version))
    print(f'''The best model has been saved with the name {model_name} version {model_details.version}
          Alias {alias_name} has been set to the best model!''')

if __name__ == '__main__':
    set_alis_for_best_model()