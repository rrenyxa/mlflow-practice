# Diabetes Prediction Project with MLflow

This project demonstrates a complete MLOps lifecycle using the Scikit-learn Diabetes dataset. It covers experiment tracking, model registration, aliasing, and metadata logging.

## Project Features
- **Experiment Tracking**: Logging hyperparameters, MSE metrics, and training epochs.
- **Artifact Logging**: Saving feature importance plots and test prediction CSVs.
- **Model Registry**: Using the **Champion** alias to manage and deploy models.
- **Model Signature**: Enforcing input/output schemas for robust inference.
- **SQLite Backend**: Local database for persistent metadata storage.

## Project Structure
- `train.py`: Main script for training and logging experiments.
- `mlflow_utils.py`: Utility to find the best run and assign the @champion alias.
- `inference.py`: Script to load the @champion model and run predictions.
- `mlflow.db`: SQLite database for MLflow metadata.
- `mlruns/`: Directory containing physical artifacts.

---

## Getting Started

### 1. Installation
This project uses **uv** for dependency management.
```bash
uv sync
```

### 2. Run the Training Pipeline
```bash
python train.py
```

### 3. Register the Best Model
```bash
python mlflow_utils.py
```

### 4. Launch MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
> **Access the UI at:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Useful MLflow Commands

### Server Management
- **Start server**: `mlflow server --backend-store-uri sqlite:///mlflow.db`
- **Cleanup**: `mlflow gc --backend-store-uri sqlite:///mlflow.db`

### Model Inference
```python
import mlflow
model = mlflow.pyfunc.load_model("models:/Diabetes_model@champion")
```

---

## Model Signature
The model includes a formal **Signature** logged during training:
- **Inputs**: 10 baseline variables (age, sex, bmi, bp, s1-s6).
- **Outputs**: Quantitative measure of disease progression.