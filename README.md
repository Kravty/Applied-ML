# Applied MLOps: Orchestration & Artifact Tracking

This repository contains modular templates and demonstrations for structuring Machine Learning pipelines. It focuses on the core components of MLOps: experiment tracking, configuration management, and reproducible artifact logging.

### Relevance to AI Security & LLMSecOps
While this repository demonstrates foundational ML engineering, these MLOps principles are directly applicable to AI Security workflows. Securing AI systems requires strict pipeline provenance, reproducible evaluation runs (e.g., for automated red-teaming), and organized artifact tracking (managing weights, datasets, and telemetry). Tools like MLflow and Hydra provide the infrastructure necessary to monitor model behavior and deployment securely.

### Contents

This repository is organized into focused demonstrations. Each component can be run independently to understand how these tools integrate into a PyTorch workflow.

| Demonstration | Core Focus | Technologies Used |
| :--- | :--- | :--- |
| **1. [Experiment Tracking](MLflow_PyTorch.ipynb)** | Logging metrics, parameters, and model artifacts for reproducibility. | MLflow, PyTorch |
| **2. [Hyperparameter Optimization](Optuna_PyTorch.ipynb)** | Automated search spaces and efficient trial pruning. | Optuna, PyTorch |
| **3. [Integrated Pipeline](./Hydra_MLflow_Optuna)** | Combining configuration management with tracking and optimization. | Hydra, MLflow, Optuna, PyTorch |

### Usage

* **.ipynb files:** Click on the 'Open In Colab' badge at the top of the notebooks to run them directly in Google Colab.
* **Integrated Pipeline Directory:** Navigate to the `Hydra_MLflow_Optuna` folder and follow the execution instructions in its specific `README.md` to run the orchestrated scripts locally.
