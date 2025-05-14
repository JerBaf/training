# training
Small library for generic GNN training.

## Concept

The library provides a set of modular classes that, given a dataset, will provide several functionalities:

- Fold processing in Pytorch Geometric format for cross-validation
- Hyper-parameter grid search with cross-validation training
- Metric logging on WandB

Each classes (or module) has a default implementation that can be overriden. The default implementation cannot work
hadhock due to dataset specifities. You will need to implement a few custom functions as indicated in the corresponding
section in the README.

## Module Structure

### Config Parser

The `Config Parser` main goal is to parse and validate the config file provided by the user. It will check some key properties, such as 
consistency between `value` and `values` keywords, path existance, etc. This requires no modification by the user for regular usage.

### Fold Manager

The `Fold Manager` is in charge of creating the fold structure for cross-validation and process the raw data to convert it from `.csv` files to Pytorch Geometric graphs in `.pt` format. Check the following [webpage](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets) if you want more information on standard practice for Pytorch Geometric Dataset implementation.

### Logger

The `Logger` module handles the logging of metrics using [Weights and Biases (WandB)](https://wandb.ai/). It already implements logging of key metrics for classification and regression tasks. The default implementation requires no modification by the user, except if the task is outside of the initial scope of classification and regression.

### Metric Calculator

The `Metric Calculator` has two main purposes. The first is to compute the loss to optimize during model training. The second is to compute all the metrics associated to the given taks.

### Trainer

The `Trainer` module is the core of the library. It handles the training procedure per fold and orchestrate the metrics computation and logging. The default implementation has a training loop already implemented but it may not suit all cases.

### Grid Search

The `Grid Search` module handles the hyperparameter selection and is the only component of the pipeline that needs to be instantiated by the user. Then, a single call to its `launch()` function will perform the cross-validation using the given config file.

## Usage

### Config

The config file provides all the parameters used to perform the training and, overall, the hyperparameter gridsearch.
It follows a similar architecture as the one employed in WandB. Parameters can be defined by two ways:

- With the `value` keyword: The value for this parameter will be fixed across the different sweeps instanciated by the Grid Search.
- With the `values` keyword: Indicate that the parameter is an hyperparameter to be tuned. The Grid Search will iterate over the different values provided.

Among the different constraints on the parameters of the Grid Search, some of them are mandatory and can only be assigned with `value` keyword. Check the table beneath to have a list of the constraints on each mandory keywords. Furthermore, an example `config.yaml` is provided with some comments to help you in constructing your own config file.

| Parameter            | Description                                         | Mandatory              | Value Only |
| :----------------    | :------                                             | :----:                 | :----:     |
| `batch_size`         |  Integer indicating the size of each data batch     | `True`                 | `False`    |
| `class_names`        |  List of names for each class                       | Only if classification | `True`     |
| `epochs`             |  Number of epochs for model training                | `True`                 | `False`    |
| `device`             |  Device to which the data and model will be cast    | `True`                 | `True`     |
| `fold_root_path`     |  Path to the folder where the folds will be stored  | `True`                 | `True`     |
| `lr`                 |  Float for learning rate of optimizer               | `True`                 | `False`    |
| `metric_list`        |  List of metrics to register on WandB               | `True`                 | `True`     |
| `num_classes`        |  Total number of classes                            | Only if classification | `True`     |
| `num_folds`          |  Number of folds for cross-validation               | `True`                 | `True`     |
| `raw_data_path`      |  Path to the folder where the raw dataset is stored | `True`                 | `True`     |
| `save_path`          |  Path to the folder where the models will be saved  | `True`                 | `True`     |
| `save_mode`          | 'min' or 'max', depending on the target metric.     | `True`                 | `True`     |
| `seed`               |  Integer to control reproducibility                 | `True`                 | `False`    |
| `target_metric`      |  Metric used to assess which model is best          | `True`                 | `True`     |
| `task_type`          |  Either regression or classification                | `True`                 | `True`     |



### Data

The library requires a specific data structure prior to the pipeline. More specifically, the `Fold_Manager` module expects that
the initial raw dataset (located under the `raw_data_path` in the config) consists of `.csv` files with as prefix the id of the
graph. Furthermore, the file storing the mapping between ids and labels (located under the `label_path` in the config) should be a `.csv`
with at least two columns: `[id,label]`. The id column must match the file ids under the `raw_data_path` folder.

### Mandatory functions to implement

The library cannot work out-of-the-box in its default state. The user needs to implement some key functions that highly depend on the dataset and downstream tasks. Here is the list of the functions that need to be implement for a regular behaviour:

- `Fold Manager`
    - `get_fold_dataset`: This function should return the training and validation dataset for the given fold path. Uses you custom Pytorch Geometric Dataset implementation there.
- `Metric Calculator`
    - `compute_loss`: Implement here the loss function that you wish to optimize through training. It should output a single tensor with the loss value and computation graph.
- `Trainer`:
    - `create_model`: This function should output the model (`torch.nn.Module`) that you wish to train.
    - `get_dataloaders`: Analogously to the `get_fold_dataset` function, this one should output the training and validation Pytorch Geometric Dataloaders. You can specify here the parameters for loaders that suits your application.

### Standard practice

