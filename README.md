# training
Small library for generic GNN training.

### Concept

The library provides a set of modular classes that, given a dataset, will provide several functionalities:

- Fold processing in Pytorch Geometric format for cross-validation
- Hyper-parameter grid search with cross-validation training
- Metric logging on WandB

Each classes (or module) has a default implementation that can be overriden. The default implementation cannot work
hadhock due to dataset specifities. You will need to implement a few custom functions as indicated in the corresponding
section in the README.

### Usage

##### Config

The config file provides all the parameters used to perform the training and, overall, the hyperparameter gridsearch.
It follows a similar architecture as the one employed in WandB. Parameters can be defined by two ways:

- With the `value` keyword: The value for this parameter will be fixed across the different sweeps instanciated by the Grid Search.
- With the `values` keyword: Indicate that the parameter is an hyperparameter to be tuned. The Grid Search will iterate over the different values provided.

Among the different constraints on the parameters of the Grid Search, some of them are mandatory and can only be assigned with `value` keyword. Check the table beneath to have a list of the constraints on each mandory keywords. Furthermore, an example `config.yaml` is provided with some comments to help you in constructing your own config file.

| Parameter            | Description                                         | Mandatory              | Value Only |
| :----------------    | :------                                             | :----:                 | :----:     |
| `class_names`        |  List of names for each class                       | Only if classification | `True`     |
| `device`             |  Device to which the data and model will be cast    | `True`                 | `True`     |
| `fold_root_path`     |  Path to the folder where the folds will be stored  | `True`                 | `True`     |
| `num_classes`        |  Total number of classes                            | Only if classification | `True`     |
| `num_folds`          |  Number of folds for cross-validation               | `True`                 | `True`     |
| `raw_data_path`      |  Path to the folder where the raw dataset is stored | `True`                 | `True`     |
| `save_path`          |  Path to the folder where the models will be saved  | `True`                 | `True`     |
| `metric_list`        |  List of metrics to register on WandB               | `True`                 | `True`     |
| `save_mode`          | 'min' or 'max', depending on the target metric.     | `True`                 | `True`     |
| `target_metric`      |  Metric used to assess which model is best          | `True`                 | `True`     |
| `task_type`          |  Either regression or classification                | `True`                 | `True`     |
| `batch_size`         |  Integer indicating the size of each data batch     | `True`                 | `False`    |
| `epochs`             |  Number of epochs for model training                | `True`                 | `False`    |
| `lr`                 |  Float for learning rate of optimizer               | `True`                 | `False`    |
| `seed`               |  Integer to control reproducibility                 | `True`                 | `False`    |


##### Data

The library requires a specific data structure prior to the pipeline. More specifically, the `Fold_Manager` module expects that
the initial raw dataset (located under the `raw_data_path` in the config) consists of `.csv` files with as prefix the id of the
graph. Furthermore, the file storing the mapping between ids and labels (located under the `label_path` in the config) should be a `.csv`
with at least two columns: `[id,label]`. The id column must match the file ids under the `raw_data_path` folder.

##### Mandatory functions to implement

##### Standard practice

### Module Structure

##### Config Parser

Its main goal is to parse and validate the config file provided by the user. It will check some key properties, such as 
consistency between `value` and `values` keywords, path existance, etc. This requires no modification by the user for regular usage.

##### Fold Manager

##### Logger

##### Metric Calculator

##### Trainer

##### Grid Search