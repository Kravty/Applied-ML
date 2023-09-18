## Hydra & MLflow & Optuna & PyTorch

### Repository description
This repository is for demonstraition purposes of [Hydra](https://hydra.cc/) capabilities and how to combine it with experiment tracking ([MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)) and automatic hyperparameters tuning ([Optuna](https://optuna.org/)). For training [PyTorch](https://pytorch.org/) is used. All those frameworks are open-source and can be used free of charge for commercial use ([Hydra](https://github.com/facebookresearch/hydra/blob/main/LICENSE) & [Optuna](https://github.com/optuna/optuna/blob/master/LICENSE): MIT licence, [MLflow](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt): Apache License 2.0, [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE): BSD-3 licence). 

#

### Installation

Make sure you have `Python>=3.6` installed on your machine. You can install required libraries using:

    pip install -r requirements.txt

Moreover, you will need PyTorch installed. Due to different cuda version requirements, I haven't included it in *requirements.txt*. To install PyTorch, follow the instructions from the PyTorch website:
* [Current PyTorch version](https://pytorch.org/get-started/locally)
* [Previous versions](https://pytorch.org/get-started/previous-versions/) (if you have an older cuda version)

### Colab

If you would like to run this repository but don't have GPU or simply don't want to pull it to your PC, then you can use the Colab link below, where the repository is pulled and run.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/Hydra_MLflow_Optuna.ipynb)


#

### Parameters in ML projects
There are many parameters in the Machine Learning projects, and storing them as hard-coded inside Python scripts is not the best idea. This problem is especially visible when the project is getting bigger and bigger or when the variables are declared on the fly when required. It is really hard to keep track of where some parameter should be changed or if it was overwritten in other place.

Instead of hardcoding parameter values, there are two other popular options:
* Use the parser to specify arguments from the command line (eg. via argparse)
* Use a configuration file to have all parameters in one place (eg. in YAML files)

Both parsing arguments and using configuration file has another advantage over hardcoding values inside Python scripts. When you want to share your code with someone it would be much easier for them to know where to change input parameters (even if he doesn't know Python) without diving deeply inside your code. Moreover, with configuration files or argument parsers, you can run multiple experiments with different arguments with a simple shell script instead of waiting for each run to finish and then changing the hardcoded value in the .py file.

Configuration files have this advantage over argparse in that it is easier to keep track of changes in configuration in time (for example via Version Control System (VCS)). Moreover, they are more convenient when we want to change multiple parameters and are superior in terms of deployment in different environments. On the other hand, when only one parameter is changed argparse can be easier to use.

### Hydra
<p align="center"><img src="https://raw.githubusercontent.com/facebookresearch/hydra/main/website/static/img/Hydra-Readme-logo2.svg" alt="logo" width="50%" /></p>

[Hydra](https://hydra.cc/) is open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

The key featres of Hydra are:
* Hierarchical configuration composable from multiple sources
* Configuration can be specified or overridden from the command line
* Dynamic command line tab completion
* Run your application locally or launch it to run remotely
* Run multiple jobs with different arguments with a single command

Hydra supports Linux, Mac, and Windows.

#

### Datasets

To demonstrate the flexibility of Hydra, two datasets will be used for demonstration:
* [Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition) - Classification task: Facial expression recognition (people)
* [🐶Pet's Facial Expression Image Dataset😻](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) - Classification task: Facial expression recognition (pets)

[🐶Pet's Facial Expression Image Dataset😻](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) is already split and stored in right the format for torchvision ImageFolder, thus the implementation is pretty straight forward. 

[Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition) is not split and require building a Custom Dataset object.

### Models

For training pre-trained models from [torchvision](https://pytorch.org/vision/stable/index.html) will be used:
* [ConvNeXt](https://arxiv.org/abs/2201.03545)
* [EfficientNetV2](https://arxiv.org/abs/2104.00298)

### Training

A few ways to trigger model training are possible:

* Using parameters specified in config files:

        python main.py

* Using parameters specified in config files and overriding part of them from CLI:

        python main.py 'param_name=new_param_value'

* Using Optuna to automatically choose best hyperparameters in several runs:

        python main.py --multirun

Warning! If you have cloned this repository, remove subdirectories of mlruns to avaid PermissionError.


### Results
Results of runs are stored in the mlruns folder so that, can be easily viewed locally using MLflow in the repository directory:
```console
foo@bar:~$ mlflow ui
INFO:waitress:Serving on http://127.0.0.1:5000
```

When the command is run, results are depicted in MLflow UI [under this address](http://127.0.0.1:5000).

The goal of this repository is to demonstrate the capabilities of the software, not to achieve state-of-the-art (SOTA) results on the dataset. Therefore, small and efficient models were used instead of the best performing ones.

### Structure of repository
```
└── Hydra & MLflow & Optuna
    ├── conf (configuration files)
    │   ├── augmentations
    │   │   └── default_augmentations.yaml
    │   ├── dataset
    │   │   ├── people_facial_expression.yaml
    │   │   └── pets_facial_expression.yaml
    │   ├── optimizer
    │   │   ├── adam.yaml
    │   │   ├── adamw.yaml
    │   │   └── sgd.yaml
    │   ├── scheduler
    │   │   ├── cosine_annealing_lr.yaml
    │   │   └── step_lr_scheduler.yaml
    │   └── config.yaml
    ├── data (directory with raw data)
    │   ├── people_facial_expression (kaggle copyrights)
    │   └── pets_facial_expression (kaggle copyrights)  
    ├── demo (directory for colab demo)
    │   └──  hydra_mlflow_optuna.ipynb
    ├── src (source code of used functions)
    │   ├── dataloader.py
    │   ├── dataset.py
    │   ├── models.py
    │   ├── tracking.py
    │   └── train.py
    ├── main.py
    └── README.md
```