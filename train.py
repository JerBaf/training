# Packages
from functools import partial
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, roc_auc_score
import torch
from torch.nn import Softmax
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Optional
import wandb
from wandb.sdk.wandb_config import Config
import yaml

class Config_Wrapper():
    def __init__(self,config):
        self.config = config
        for config_key, value in config.items():
            # Dictionnary
            if type(value) == dict:
                if 'values' in value:
                    setattr(self,config_key,value['values'][0])
                else:
                    setattr(self,config_key,value['value'])
            # WandB
            else:
                setattr(self,config_key,value)

    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({items})"

class Config_Parser():
    """ Parse and verify integrity of a config file. """

    CLASSIFICATION_KEYS = {'class_names','num_classes'}
    MANDATORY_KEYS = {'device','fold_root_path','label_path','num_folds','raw_data_path',
                      'save_path','target_dim','metric_list','save_mode','target_metric',
                      'task_type','batch_size','epochs','lr','seed'}
    PATH_KEYS = {'fold_root_path','label_path','raw_data_path','save_path'}
    VALUE_ONLY_KEYS = CLASSIFICATION_KEYS.union(PATH_KEYS).union({'device','num_folds','metric_list','save_mode','target_dim','target_metric','task_type'})
    
    def __init__(self,config_path:str):
        self.config_path = config_path

    def parse_config(self) -> dict:
        """ Parse and validate the config file given at initialization. """
        # Parse config
        with open(self.config_path,'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(f"Impossible to parse config in {self.config_path}") from exc
        # Validate config integrity
        try:
            self.validate_config(config)
            return Config_Wrapper(config)
        except Exception as exc:
            raise ValueError(f"Invalid config configuration") from exc
            
    def validate_config(self,config:dict):
        """ Verify the integrity of a config file. """
        # Check required entries
        missing_entries = self.MANDATORY_KEYS.difference(set(config.keys()))
        if missing_entries:
            raise ValueError(f'The config is missing the following entries: {missing_entries}')
        classification_missing_entries = self.CLASSIFICATION_KEYS.difference(set(config.keys()))
        if config['task_type'] == 'classification' and classification_missing_entries:
            raise ValueError(f'The config is missing the following entries: {classification_missing_entries}')
        # Check validaty of 'values' entries
        for key, values_dict in config.items():
            if 'values' in values_dict and type(values_dict['values']) != list:
                raise ValueError(f"The config entry {key} has 'values' specified but no list of parameters is given.")
        # Check value entry
        self.validate_value_entry(config)
        # Check path is correct
        self.validate_path(config)
        # Check label file 
        self.validate_labels(config)
    
    def validate_labels(self,config:dict):
        """ Validate the format of the label files. """
        label_files = config['label_path']['values'] if 'values' in config['label_path'] else [config['label_path']['value']]
        for l in label_files:
            try: 
                label_df = pd.read_csv(l)
                label_df.set_index('id')
                label_df['label'].value_counts()
            except:
                raise ValueError(f'Impossible to read and parse correctly the label file {l}. Please check format and the corresponding README section.')

    def validate_path(self,config:dict):
        """ Raise an Exception if the provided path is not valid. """
        for path_key in list(self.PATH_KEYS):
            if 'values' in config[path_key]:
                raise ValueError(f"The config file has multiple paths for entry {path_key}. Only support one path at a time with 'value' keyword.")
            path = config[path_key]['value']
            if not os.path.exists(path) and not (path_key == 'save_path' and path == ''):
                raise ValueError(f'The config provided non existing path for entry {path_key}: {path}. Please create it prior to processing.')
        
    def validate_value_entry(self,config:dict):
        """ Verify that all entries requiring value only are valid. """ 
        for key in list(self.VALUE_ONLY_KEYS):
            if not 'value' in config[key]:
                raise ValueError(f"The config has entry {key} without a 'value' type of item. Please provide only value only item.")

class Fold_Manager():
    """ Create and pre-process the folds prior to cross-validation. """

    def __init__(self,config:Config_Wrapper):
        self.config = config
    
    def is_created(self):
        """ Boolean indicator if the fold structure has been created. """
        fold_root_path = self.config.fold_root_path
        return os.path.exists(os.path.join(fold_root_path,'0'))

    def fold_creation(self,fold_root_path:str,raw_data_path:str):
        """ Split training data in folds with stratification on labels for cross-validation. """
        # Sanity check
        if self.is_created():
            return
        # List files
        file_dict = dict([(int(f.removesuffix('.csv')),os.path.join(raw_data_path,f)) for f in os.listdir(raw_data_path)])
        # Retrieve label mapping
        label_path = self.config.label_path
        labels_df = pd.read_csv(label_path).set_index('id')
        # Regroup files per label
        stratified_files = dict()
        for file_id in file_dict.keys():
            file_label = labels_df.loc[file_id]['label']
            if file_label not in stratified_files:
                stratified_files[file_label] = []
            stratified_files[file_label].append(file_id)
        # Shuffle label-associated file lists
        generator = np.random.default_rng(self.config.seed)
        for label in stratified_files.keys():
            generator.shuffle(stratified_files[label])
        # Split per fold
        num_folds = self.config.num_folds
        for i in range(num_folds):
            fold_train_ids = []
            fold_val_ids = []
            for _, file_ids in stratified_files.items():
                label_step_size = int(np.ceil(len(file_ids)/num_folds))
                fold_train_ids.extend(file_ids[0:i*label_step_size]+file_ids[(i+1)*label_step_size:])
                fold_val_ids.extend(file_ids[i*label_step_size:(i+1)*label_step_size])
            # Retrieve files from label stratification
            fold_train_files = []
            fold_val_files = []
            for file_id in fold_train_ids:
                fold_train_files.append((file_id,file_dict[file_id]))
            for file_id in fold_val_ids:
                fold_val_files.append((file_id,file_dict[file_id]))
            # Copy files to directory structure
            fold_files = {'train':fold_train_files,'val':fold_val_files}
            fold_path = os.path.join(fold_root_path,str(i))
            os.mkdir(fold_path)
            for split_type in ['train','val']:
                split_path = os.path.join(fold_path,split_type)
                split_files_path = os.path.join(split_path,'raw')
                split_processed_path = os.path.join(split_path,'processed')
                os.mkdir(split_path)
                os.mkdir(split_files_path)
                os.mkdir(split_processed_path)
                for file_name,file_path in fold_files[split_type]:
                    shutil.copyfile(file_path,os.path.join(split_files_path,f'{file_name}.csv'))

    def process_folds(self):
        """ Process each folds individually. """
        fold_root_path = self.config.fold_root_path
        label_path = self.config.label_path
        raw_data_path = self.config.raw_data_path
        if not self.is_created():
            self.fold_creation(fold_root_path,raw_data_path)
        # Process files
        for fold_id in [d for d in os.listdir(fold_root_path) if os.path.isdir(os.path.join(fold_root_path,d))]:
            fold_path = os.path.join(fold_root_path,fold_id)
            _, _ = self.__class__.get_fold_dataset(fold_path,label_path)
     
    @classmethod
    def get_fold_dataset(cls,fold_path:str,label_path:str) -> tuple:
        """ Retrieve the train and validation dataset for a given fold. """
        return None, None
   
class Logger():
    """ Class to log the results through wandb. """

    CLASS_ASSOCIATED_METRICS = {'recall','precision'}

    def __init__(self,config:Config_Wrapper):
        self.config = config
        self.metric_list = config.metric_list
        self.fold_metric_dict = self.create_metric_dict()

    def create_metric_dict(self) -> dict:
        """ Create a dictionnary to register metrics. """
        metric_list = self.get_metric_list()
        return dict(zip(metric_list,[list() for _ in range(len(metric_list))]))
    
    def get_metric_list(self) -> list:
        """ Get the list of metrics. """
        full_metric_list = []
        for m in self.metric_list:
            if m in self.CLASS_ASSOCIATED_METRICS:
                for c in self.config.class_names:
                    full_metric_list.extend([f'train_{m}_{c}',f'val_{m}_{c}'])
            else:
                full_metric_list.extend([f'train_{m}',f'val_{m}'])
        return full_metric_list
     
    def log_confusion_matrix(self,pred_dict:dict):
        """ Register the confusion matrices on wandb. """
        if self.config.task_type == 'classification':
            for split in ['train','val']:
                y_true, y_pred = pred_dict[split]['y_true'].argmax(dim=1).detach().cpu().numpy(), pred_dict[split]['y_pred'].argmax(dim=1).detach().cpu().numpy()
                wandb.log({f"{split}_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                                    y_true=y_true, preds=y_pred,
                                                    class_names=self.config.class_names)})

    def update_metrics(self,metric_values:dict,prefix:str):
        """ Update and register the metrics on wandb. """
        metric_dict = dict()
        # Metrics Registration
        for metric_name, metric_value in metric_values.items():
            metric_dict[f'{prefix}_{metric_name}'] = metric_value
            self.fold_metric_dict[f'{prefix}_{metric_name}'].append(metric_value)
        wandb.log(metric_dict)

class Metric_Calculator():
    """ Compute the metrics before passing them to optimizer and logger. """

    def __init__(self,config:Config_Wrapper):
        self.config = config

    def compute_loss(self,y_true:torch.Tensor,y_pred:torch.Tensor) -> torch.Tensor:
        """ Compute the Pytorch loss for backpropagation. """
        loss = None
        return loss

    def compute_loss_and_metrics(self,pred_dict:dict,split:str) -> dict:
        """ Compute loss and task-associated metrics. """
        # Compute task-associated metrics
        y_true, y_pred = pred_dict[split]['y_true'], pred_dict[split]['y_pred']
        if self.config.task_type == 'classification':
            metric_dict = self.compute_classification_metrics(y_true,y_pred)
        else:
            metric_dict = self.compute_regression_metric(y_true,y_pred)
        # Compute loss
        loss = self.compute_loss(y_true,y_pred).cpu().detach()
        metric_dict['loss'] = loss
        return metric_dict

    def compute_classification_metrics(self,y_true:torch.Tensor,y_pred:torch.Tensor) -> dict:
        """ Compute classification related metrics. """
        # Labels and predictions formatting
        true_np = y_true.detach().cpu().numpy()
        pred_np = y_pred.detach().cpu().numpy()
        if true_np.ndim > 1:
            labels = true_np.argmax(axis=1)
        else:
            labels = true_np.astype(int)
        probs = Softmax(dim=-1)(torch.from_numpy(pred_np)).numpy()
        preds = probs.argmax(axis=1)
        # Retrieve label list
        label_list = self.config.class_names
        # Metrics computation
        metric_dict = dict()
        metrics = classification_report(labels,preds,target_names=label_list,output_dict=True)
        # Compute AUC
        if len(label_list) == 2:
            probs = probs[:,1]
            auc = roc_auc_score(labels,probs)
        else:
            auc = roc_auc_score(labels,probs,multi_class='ovo',labels=range(len(label_list)))    
        # Retrieve averages
        macro_f1 = metrics['macro avg']['f1-score']
        weighted_f1 = metrics['weighted avg']['f1-score']
        # Metrics Registration
        metric_dict['auc'] = auc
        metric_dict['macro_f1'] = macro_f1
        metric_dict['weighted_f1'] = weighted_f1
        # Recall and precision registration
        for label_id in label_list:
            for metric_type in ['recall','precision']:
                metric_value = metrics[label_id][metric_type]
                metric_dict[f'{metric_type}_{label_id}'] = metric_value
        return metric_dict

    def compute_regression_metric(self,y_true:torch.Tensor,y_pred:torch.Tensor) -> dict:
        """ Compute regression related metrics. """
        # Labels and predictions formatting
        labels = y_true.detach().cpu().numpy().ravel()
        preds  = y_pred.detach().cpu().numpy().ravel()
        # Metric computation
        metric_dict = dict()
        mse = mean_squared_error(labels,preds)
        mae = mean_absolute_error(labels,preds)
        # Metrics Registration
        metric_dict['mse'] = mse
        metric_dict['mae'] = mae
        return metric_dict

class Trainer():
    """ Class to perform the training for one given fold. """

    def __init__(self,config:Config_Wrapper,fold_id:int,fold_manager_cls=Fold_Manager,logger_cls=Logger,metric_calculator_cls=Metric_Calculator):
        # Attributes
        self.config = config
        self.device = torch.device(config.device)
        self.fold_id = fold_id
        self.fold_manager_cls = fold_manager_cls
        self.metric_calculator = metric_calculator_cls(config)
        self.logger = logger_cls(config)

    def create_balanced_sampler(self,dataset) -> WeightedRandomSampler:
        """ Create a balanced sampler for training based on class distribution."""
        labels = dict([(i,0) for i in range(self.config.num_classes)])
        # Compute label distribution
        for idx in range(len(dataset)):
            data = dataset[idx]
            label = torch.argmax(data.y).item()
            labels[label] += 1
        # Compute weights
        weights = dict([(k,len(dataset)/v) for k,v in labels.items()])
        weight_list = []
        for idx in range(len(dataset)):
            data = dataset[idx]
            label = torch.argmax(data.y).item()
            weight_list.append(weights[label])
        return WeightedRandomSampler(weight_list,len(dataset))

    def create_model(self) -> torch.nn.Module:
        """ Create an instance of the model defined in the config file. """
        # Often call get_input_dim and then instantiate the model
        return None

    def get_datasets(self) -> tuple :
        """ Retrieve the train and validation dataset. """
        fold_path = os.path.join(self.config.fold_root_path,str(self.fold_id))
        label_path = self.config.label_path
        return self.fold_manager_cls.get_fold_dataset(fold_path,label_path)

    def get_dataloaders(self,generator:torch.Generator) -> tuple:
        """ Retrieve the train and validation dataloaders for a given fold. """
        return None, None

    def get_input_dim(self,dataset:Dataset) -> int:
        """ Get the input dimension of samples given to the model. """
        example_data_point = dataset[0]
        input_dim = example_data_point.x.shape[1]
        return input_dim

    def is_best_model(self) -> bool:
        """ Boolean indicator if the last epoch resulted in the best trained model so far. """
        fold_metric_dict = self.logger.fold_metric_dict
        target_metric = f'val_{self.config.target_metric}'
        if self.config.save_mode == 'min':
            metric_condition = fold_metric_dict[target_metric][-1] <= np.min(fold_metric_dict[target_metric])
        else:
            metric_condition = fold_metric_dict[target_metric][-1] >= np.max(fold_metric_dict[target_metric])
        return metric_condition
        
    def train(self):
        """ Perform full training for one fold in cross-validation. """
        # Initialization
        generator = self.set_seed()
        train_loader, val_loader = self.get_dataloaders(generator)
        model = self.create_model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        # Model training
        dataloader_dict = {'train':train_loader,'val':val_loader}
        for epoch in range(self.config.epochs):
            pred_dict = {'train':{'y_true':[],'y_pred':[]},
                         'val':{'y_true':[],'y_pred':[]}}
            for split, dataloader in dataloader_dict.items():
                # Set training mode
                if split == 'train':
                    model.train()
                    self.train_one_epoch(dataloader,epoch,model,optimizer,pred_dict,split)
                else:
                    model.eval()
                    with torch.no_grad():
                        self.train_one_epoch(dataloader,epoch,model,optimizer,pred_dict,split)
            # Save if save path provided
            self.save_model(model,pred_dict)
        # Release memory
        del model

    def train_one_epoch(self,dataloader,epoch:int,model:torch.nn.Module,optimizer,pred_dict:dict,split:str):
        """ Perfrom evaluation/training for the given split. """
        for batch in dataloader:
            # Forward Pass
            batch = batch.to(self.device)
            y_true, y_pred = batch.y.reshape((-1,self.config.num_classes)), model(batch)
            loss = self.metric_calculator.compute_loss(y_true,y_pred)
            # Update using the gradients
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()
            # Update running list of targets/preds
            pred_dict[split]['y_true'].append(y_true)
            pred_dict[split]['y_pred'].append(y_pred)
        # Register metrics
        pred_dict[split]['y_true'] = torch.cat(pred_dict[split]['y_true'])
        pred_dict[split]['y_pred'] = torch.cat(pred_dict[split]['y_pred'])
        metric_values = self.metric_calculator.compute_loss_and_metrics(pred_dict,split)
        self.logger.update_metrics(metric_values,split)
        
    def save_model(self,model:torch.nn.Module,pred_dict:dict):
        """ If the config allows it, save the model and register the confusion matrices. """
        if self.config.save_path != '' and self.is_best_model():
            torch.save(model,os.path.join(self.config.save_path,f'best_model_fold_{self.fold_id}.pth'))
            self.logger.log_confusion_matrix(pred_dict)
        
    def set_seed(self) -> torch.Generator:
        """ Setup reproducibility for a set of libraries. """
        seed = self.config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

class Grid_Search():
    """ Grid search object that implements cross-validation search with WandB sweeps. """
    
    def __init__(self,config_path:str,name:str,method:str='grid',config_parser_cls=Config_Parser,
                 fold_manager_cls=Fold_Manager,logger_cls=Logger,metric_calculator_cls=Metric_Calculator,
                 trainer_cls=Trainer):
        # Classes
        self.config_parser_cls = config_parser_cls
        self.fold_manager_cls = fold_manager_cls
        self.logger_cls = logger_cls
        self.metric_calculator_cls = metric_calculator_cls
        self.trainer_cls = trainer_cls
        # Attributes
        self.config_wrapper = config_parser_cls(config_path).parse_config()
        self.config = self.config_wrapper.config
        fold_manager_cls(self.config_wrapper).process_folds()
        self.method = method
        self.name = name

    def create_sweep_config(self) -> dict:
        """ Create the sweep config for the wandb grid search.
        
            :param config: Dictionnary with the parameters list and their corresponding list of value to be tested.
            :param method: Method for the search of hyperparameter, default is grid.
            
            :return: Sweep config for wandb agent in dictionnary format.
        
        """
        sweep_config = {'method': self.method}
        metric = {'name': 'Loss','goal': 'minimize'}
        sweep_config['metric'] = metric
        sweep_config['parameters'] = self.config
        return sweep_config

    def launch(self):
        """ Hyper parameter search with cross-validation for a GNN. """
        # Process to cross validation
        std_dict = dict()
        config = self.config
        for fold_id in range(config['num_folds']['value']):
            # Instantiate metric registration
            sweep_config = self.create_sweep_config()
            sweep_id = wandb.sweep(sweep_config, project=self.name)
            # Launch training
            wandb.agent(sweep_id,partial(self.training_single_fold,fold_id=fold_id,std_dict=std_dict))
        # Log standard deviations
        sweep_config = self.create_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=self.name)
        wandb.agent(sweep_id,partial(self.std_logging,std_dict=std_dict))

    def std_logging(self,config:Config=None,std_dict:Optional[dict]=None):
        """ Register the std for all metrics. """
        std_dict = {} if std_dict is None else std_dict
        # Log standard deviations
        with wandb.init(config=config):
            for metric_key, metric_values in std_dict.items():
                wandb.log({f'{metric_key}_std':np.std(metric_values)})
            
    def training_single_fold(self,config:Config=None,fold_id:int=0,std_dict:Optional[dict]=None):
        """ Training and metric registration for a single fold
            
            :param config: Wandb config for the hyperparameter search.
            :param fold_id: Id of the fold for this training.
            :param std_dict: Dictionnary to keep track of metric values for latter std computation.
        
        """
        std_dict = {} if std_dict is None else std_dict
        with wandb.init(config=config):
            config = Config_Wrapper(wandb.config)
            # Perform training
            trainer = self.trainer_cls(config,fold_id,self.fold_manager_cls,self.logger_cls,self.metric_calculator_cls)
            trainer.train()
            # Assign target specific variables
            metric_list = trainer.logger.get_metric_list()
            fold_metric_dict = trainer.logger.fold_metric_dict
            # Log global metrics
            global_metric_dict = dict()
            for metric_name, metric_values in fold_metric_dict.items():
                if metric_name in metric_list:
                    if config.task_type == 'classification':
                        if 'loss' in metric_name:
                            metric_key = f'min_{metric_name}'
                            global_metric_dict[metric_key] = np.min(metric_values)
                        else:
                            metric_key = f'max_{metric_name}'
                            global_metric_dict[metric_key] = np.max(metric_values)
                    else:
                        metric_key = f'min_{metric_name}'
                        global_metric_dict[metric_key] = np.min(metric_values)
                    # Register to standard deviation dictionnary
                    if metric_key not in std_dict:
                        std_dict[metric_key] = []
                    std_dict[metric_key].append(np.min(metric_values))
            wandb.log(global_metric_dict)
            wandb.log({'fold_id':fold_id})
 