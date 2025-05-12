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
It follows a similar architecture as the one employed in WandB. 

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