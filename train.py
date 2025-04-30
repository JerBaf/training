# Packages
from functools import partial
import numpy as np
import os
import random
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, roc_auc_score
import torch
from torch.nn import Softmax
from torch.utils.data import Dataset, WeightedRandomSampler
import wandb
from wandb.sdk.wandb_config import Config


class Metric_Calculator():
    """ Compute the metrics before passing them to optimizer and logger. """

    def __init__(self,config:Config):
        self.config = config

    def compute_loss(self,y_true,y_pred) -> torch.Tensor:
        """ Compute the Pytorch loss for backpropagation"""
        return torch.Tensor()

    def compute_loss_and_metrics(self,batch,y_pred) -> tuple:
        """ Compute loss and task-associated metrics. """
        loss, metric_dict = None, None
        return loss, metric_dict

    @classmethod
    def compute_classification_metrics(self,y_true,y_pred) -> dict:
        """ Compute classification related metrics. """
        # Labels and predictions formatting
        s_func = Softmax(dim=-1)
        labels = np.concatenate([np.copy(np.array(l.cpu())).argmax(axis=1) for l in y_true],axis=0)
        probs = np.concatenate([np.copy(np.array(s_func(p.detach().cpu()))) for p in y_pred],axis=0)
        pred = np.concatenate([np.copy(np.array(p.detach().cpu())).argmax(axis=1) for p in y_pred],axis=0)
        # Retrieve label list
        label_list = [str(i) for i in range(self.config.num_classes)] 
        # Metrics computation
        metric_dict = dict()
        metrics = classification_report(labels,pred,labels=label_list,output_dict=True)
        # Compute AUC
        if len(label_list) == 2:
            probs = probs[:,1]
            auc = roc_auc_score(labels,probs)
        else:
            auc = roc_auc_score(labels,probs,multi_class='ovo',labels=[int(l) for l in label_list])    
        # Retrieve averages
        weighted_f1 = metrics['weighted avg']['f1-score']
        macro_f1 = metrics['macro avg']['f1-score']
        # Metrics Registration
        metric_dict['auc'] = auc
        metric_dict['weighted_f1'] = weighted_f1
        metric_dict['macro_f1'] = macro_f1
        # Recall and precision registration
        for label_id in label_list:
            for metric_type in ['recall','precision']:
                metric_value = metrics[label_id][metric_type]
                metric_dict[f'{metric_type}_{label_id}'] = metric_value
        return metric_dict

    @classmethod
    def compute_regression_metric(self,y_true,y_pred) -> dict:
        """ Compute regression related metrics. """
        # Labels and predictions formatting
        labels = np.concatenate([np.copy(np.array(l.cpu())) for l in y_true],axis=0)
        preds = np.concatenate([np.copy(np.array(p.detach().cpu())) for p in y_pred],axis=0)
        # Metric computation
        metric_dict = dict()
        mse = mean_squared_error(labels,preds)
        mae = mean_absolute_error(labels,preds)
        # Metrics Registration
        metric_dict['mse'] = mse
        metric_dict['mae'] = mae
        return metric_dict

class Logger():
    """ Class to log the results through wandb. """

    def __init__(self,config:Config):
        self.config = config
        self.cross_val_dict = self.create_metric_dict()
        self.metric_list = config.metric_list

    def get_metric_list(self) -> list:
        """ Get the list of metrics. """
        full_metric_list = []
        full_metric_list.extend(['train_'+m for m in self.metric_list])
        full_metric_list.extend(['val_'+m for m in self.metric_list])
        return full_metric_list
    
    def create_metric_dict(self) -> dict:
        """ Create a dictionnary to register metrics. """
        metric_list = self.get_metric_list()
        return dict(zip(metric_list,[list() for _ in range(len(metric_list))]))
    
    def update_metrics(self,metric_values:dict,prefix:str):
        """ Update and register the metrics on wandb. """
        metric_dict = dict()
        # Metrics Registration
        for metric_name, metric_value in metric_values.items():
            metric_dict[f'{prefix}_{metric_name}'] = metric_value
            self.cross_val_dict[f'{prefix}_{metric_name}'].append(metric_value)
        wandb.log(metric_dict)

class Trainer():
    """ Class to perform cross-validation training. """

    def __init__(self,config:Config,fold_id:int):
        self.config = config
        self.device = torch.device(config.device)
        self.fold_id = fold_id
        self.metric_calculator = Metric_Calculator(config)
        self.logger = Logger(config)

    def create_balanced_sampler(self,dataset) -> WeightedRandomSampler:
        """ Create a balanced sampler for training based on class distribution."""
        labels = dict([(i,0) for i in range(self.config.num_classes)])
        # Compute label distribution
        for idx in range(len(dataset)):
            data = dataset.get(idx)
            label = torch.argmax(data[-1].y).item()
            labels[label] += 1
        # Compute weights
        weights = dict([(k,len(dataset)/v) for k,v in labels.items()])
        weight_list = []
        for idx in range(len(dataset)):
            data = dataset.get(idx)
            label = torch.argmax(data[-1].y).item()
            weight_list.append(weights[label])
        return WeightedRandomSampler(weight_list,len(dataset))

    def create_model(self) -> torch.nn.Module:
        """ Create an instance of the model defined in the config file. """
        # Often call get_input_dim and then instantiate the model
        return None

    def get_datasets(self) -> tuple :
        """ Retrieve the train and validation dataset for a given fold. """
        return None, None

    def get_dataloaders(self,generator:torch.Generator) -> tuple:
        """ Retrieve the train and validation dataloaders for a given fold. """
        return None, None

    def get_input_dim(self,dataset:Dataset) -> int:
        """ Get the input dimension of samples given to the model. """
        example_data_point = dataset.get(0)
        input_dim = example_data_point.x.shape[1]
        return input_dim

    def train(self):
        """ Perform full training for one fold in cross-validation. """
        # Initialization
        generator = self.set_seed()
        train_loader, val_loader = self.get_dataloaders(generator)
        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        # Model training
        for epoch in range(self.config.epochs):
            self.train_one_epoch(optimizer,model,train_loader,val_loader,epoch)
        # Release memory
        del model

    def train_one_epoch(self,epoch,model,optimizer,train_loader,val_loader):
        """ Train the model for a single epoch. """
        dataloader_dict = {'train':train_loader,'val':val_loader}
        # Model training
        for split, dataloader in dataloader_dict.items():
            # Set training mode
            if split == 'train':
                model.train()
            else:
                model.eval()
            # Batch loop
            for batch in dataloader:
                # Forward Pass
                preds = model.forward(batch)
                loss, metric_values = self.metric_calculator.compute_loss_and_metrics(batch,preds)
                # Update using the gradients
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()
            # Register metrics
            self.logger.update_metrics(metric_values,split)
        # Save if save path provided
        self.save_model(model)
        
    def save_model(self,model:torch.nn.Module):
        """ Save the model if the config allows it. """
        if self.config.save_path != '' and cross_val_dict[f'val_total_loss'][-1] <= np.min(cross_val_dict[f'val_total_loss']):
            torch.save(model,os.path.join(config.save_path,f'model_fold_{fold_id}_best_loss.pth'))

    def set_seed(self) -> torch.Generator:
        """ Setup reproducibility for a set of libraries. """
        seed = self.config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

# Grid Search Helpers

def std_logging(config:Config=None,std_dict:dict=dict()):
    """ Register the std for all metrics. """
    # Log standard deviations
    with wandb.init(config=config):
        for metric_key, metric_values in std_dict.items():
            wandb.log({f'{metric_key}_std':np.std(metric_values)})

def create_sweep_config(parameters_dict:dict,method:str='grid') -> dict:
    """ Create the sweep config for the wandb grid search.
    
        :param parameters_dict: Dictionnary with the parameters list and
                                their corresponding list of value to be tested.
        :param method: Method for the search of hyperparameter, default is grid.
        
        :return: Sweep config for wandb agent in dictionnary format.
    
    """
    
    sweep_config = {
    'method': method
    }
    metric = {
    'name': 'MeanSquareError',
    'goal': 'minimize'   
    }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    return sweep_config

def training_pipeline(parameters_dict:dict,project_name:str,method:str='grid'):
    """ Hyper parameter search with cross-validation for a GNN. """
    # Process to cross validation
    std_dict = dict()
    for fold_id in range(len(parameters_dict['data_paths']['values'][0])):
        # Instantiate metric registration
        sweep_config = create_sweep_config(parameters_dict,method=method)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        # Launch training
        wandb.agent(sweep_id,partial(training_single_fold,fold_id=fold_id,std_dict=std_dict))
    # Log standard deviations
    sweep_config = create_sweep_config(parameters_dict,method=method)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id,partial(std_logging,std_dict=std_dict))

def training_single_fold(config:Config=None,fold_id:int=0,std_dict:dict=dict()):
    """ Training and metric registration for a single fold
        
        :param config: Wandb config for the hyperparameter search.
        :param fold_id: Id of the fold for this training.
        :param std_dict: Dictionnary to keep track of metric values for latter std computation.
    
    """
    with wandb.init(config=config):
        config = wandb.config
        # Perform training
        trainer = Trainer(config,fold_id)
        trainer.train()
        # Assign target specific variables
        metric_list = trainer.logger.metric_list
        cross_val_dict = trainer.logger.cross_val_dict
        # Log global metrics
        global_metric_dict = dict()
        for metric_name, metric_values in cross_val_dict.items():
            if metric_name in metric_list:
                if config.categorical:
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
