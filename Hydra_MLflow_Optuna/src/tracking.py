from collections import OrderedDict
from typing import Dict, Union, Optional

# external lib imports
from hydra import utils
import mlflow
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch


# define MLflow logger
class MLflowLogger:
    """
    A class to represent logger object, which can be used to log tracking data to MLflow
    
    Args:
        cfg (DictConfig): configuration file loaded by Hydra
    
    """
    def __init__(self, cfg: DictConfig):
        mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
        self.cfg = cfg
        self.client = MlflowClient()
        # create separate experiment with dataset name to distinguish runs at each dataset
        experiment_name = cfg.dataset.name
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def save_multiple_records(self, metric_name: str, metric_values: list) -> None:
        """
        Method to log list of metrics to mlflow.

        Args:
            metric_name: name of the metric to log
            metric_values: list of metric values to log
        """
        for metric_value in metric_values:
            self.client.log_metric(self.run_id, metric_name, metric_value)
        
        return
    
    def log_params(self):
        """
        Function log params specified in config file in 'params_to_log'
        """
        for param_name, param_value in self.cfg.params_to_log.items(): 
            self.client.log_param(self.run_id, key=param_name, value=param_value)
    
    def log_to_mlflow(self, metrics: dict) -> None:
        """
        Method to log parameters and metrics to mlflow.

        Args:
            metrics: dictionary with metrics values for each key to log
        """
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, list):
                self.save_multiple_records(metric_name=metric_name, metric_values=metric_value)
            else:
                self.client.log_metric(self.run_id, metric_name, metric_value)
        # save full config as artifact
        self.client.log_dict(self.run_id, dictionary=OmegaConf.to_container(self.cfg), artifact_file="config.yaml")
        # save key params for comparison
        self.log_params()

        return
    
    def log_model_to_mlflow(self, state_dict: Dict[str, Union[int, OrderedDict, dict]], model: Optional[torch.nn.Module]=None) -> None:
        """
        Function to log parameters and metrics to mlflow.

        Args:
            model: configuration file loaded by Hydra
        """
        with mlflow.start_run(self.run_id):
            mlflow.pytorch.log_model(model, 'model')
        # with mlflow.start_run(experiment_id=self.experiment_id):
            # # mlflow.pytorch.log_state_dict(state_dict, artifact_path=cfg.paths.model_checkpoint)
            # mlflow.pytorch.log_model(model, "model")
        
        return
    
    def load_logged_model(self) -> torch.nn.Module:
        """
        Function load model logged in MLflow
        """
        model_uri = f"runs:/{self.run_id}/model"
        model = mlflow.pytorch.load_model(model_uri=model_uri)

        return model
    
    



# def set_up_mlflow(cfg):
#     # set up path where tracking data should be saved (hydra can change cwd) 
#     mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
#     # create separate experiment with dataset name to distinguish runs at each dataset
#     experiment = mlflow.set_experiment(cfg.dataset.name)
#     experiment.experiment_id
    


# def save_multiple_records(metric_name: str, metric_values: list) -> None:
#     """
#     Function to log list of metrics to mlflow.

#     Args:
#         metric_name: name of the metric to log
#         metric_values: list of metric values to log
#     """
#     for metric_value in metric_values:
#         mlflow.log_metric(metric_name, metric_value)

#     return


# def log_to_mlflow(cfg: DictConfig, metrics: dict) -> None:
#     """
#     Function to log parameters and metrics to mlflow.

#     Args:
#         cfg: configuration file loaded by Hydra
#         metrics: dictionary with metrics values for each key to log
#     """


#     with mlflow.start_run(experiment_id=experiment.experiment_id):
#         # # log parameters
#         # for param_name, param_value in cfg.params.items():
#         #     mlflow.log_param(param_name, param_value)
#         # # log model parameters
#         # for model_param_name, model_param_value in cfg.model.items():
#         #     mlflow.log_param(model_param_name, model_param_value)
#         # # log remaining parameters specified in defaults
#         # mlflow.log_param("optimizer", cfg.optimizer)
#         # mlflow.log_param("scheduler", cfg.scheduler)
       
#         # log metrics
#         for metric_name, metric_value in metrics.items():
#             if isinstance(metric_value, list):
#                 save_multiple_records(metric_name=metric_name, metric_values=metric_value)
#             else:
#                 mlflow.log_metric(metric_name, metric_value)
        
#         # save full config as artifact
#         mlflow.log_dict(dictionary=OmegaConf.to_container(cfg), artifact_file="config.yaml")

#     return


# def log_model_to_mlflow(cfg: DictConfig, state_dict: Dict[str, Union[int, OrderedDict, dict]], model=None) -> None:
#     """
#     Function to log parameters and metrics to mlflow.

#     Args:
#         cfg: configuration file loaded by Hydra
#     """

#     # set up path where tracking data should be saved (hydra can change cwd) 
#     mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
#     # create separate experiment with dataset name to distinguish runs at each dataset
#     experiment = mlflow.set_experiment(cfg.dataset.name)
#     with mlflow.start_run(experiment_id=experiment.experiment_id):
#         # mlflow.pytorch.log_state_dict(state_dict, artifact_path=cfg.paths.model_checkpoint)
#         mlflow.pytorch.log_model(model, "model")

#     return
