# Packages
from dataset import get_datasets, TemporalDataLoader
from functools import partial
from models.vae_temporal import categorical_loss_fn, VAE
import numpy as np
import os
import random
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
import torch
from torch.nn import Softmax
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import wandb
from wandb.sdk.wandb_config import Config


class Metric_Calculator():
    """ Compute the metrics before passing them to optimizer and logger. """

    def __init__(self,config:Config):
        self.config = config

    def compute_loss(self,y_true,y_pred) -> torch.Tensor:
        """ Compute the Pytorch loss for backpropagation"""
        return torch.Tensor()

    def compute_loss_and_metrics(self,y_true,y_pred) -> tuple:
        """ Compute loss and task-associated metrics. """
        loss, metric_dict = None, None
        return loss, metric_dict

    @classmethod
    def compute_classification_metrics(self,y_true,y_pred) -> dict:
        """ Compute classification related metrics. """
        # Labels and predictions formatting
        s_func = Softmax(dim=-1)
        labels = np.concatenate([np.copy(np.array(l.cpu())).argmax(axis=1) for l in labels],axis=0)
        probs = np.concatenate([np.copy(np.array(s_func(p.detach().cpu()))) for p in pred],axis=0)
        pred = np.concatenate([np.copy(np.array(p.detach().cpu())).argmax(axis=1) for p in pred],axis=0)
        # Retrieve label list
        label_list = [str(i) for i in range(config.num_classes)] 
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
        metric_dict[f'{prefix}_loss'] = loss
        cross_val_dict[f'{prefix}_loss'].append(loss)
        metric_dict[f'{prefix}_auc'] = auc
        cross_val_dict[f'{prefix}_auc'].append(auc)
        metric_dict[f'{prefix}_weighted_f1'] = weighted_f1
        cross_val_dict[f'{prefix}_weighted_f1'].append(weighted_f1)
        metric_dict[f'{prefix}_macro_f1'] = macro_f1
        cross_val_dict[f'{prefix}_macro_f1'].append(macro_f1)
        # Recall and precision registration
        for label_id in label_list:
            for metric_type in ['recall','precision']:
                metric_value = metrics[label_id][metric_type]
                metric_dict[f'{prefix}_{metric_type}_{label_id}'] = metric_value
                cross_val_dict[f'{prefix}_{metric_type}_{label_id}'].append(metric_value)
        wandb.log(metric_dict)
        # Update confusion matrix 
        if macro_f1 >= np.max(cross_val_dict[f'{prefix}_macro_f1']):
            cross_val_dict[f'{prefix}_best_model_labels'] = labels
            cross_val_dict[f'{prefix}_best_model_preds'] = pred

    @classmethod
    def compute_regression_metric(self,y_true,y_pred) -> dict:
        """ Compute regression related metrics. """


    

class Logger():
    """ Class to log the results through wandb. """

    def __init__(self,config:Config,metric_list:list):
        self.config = config
        self.cross_val_dict = self.create_metric_dict()
        self.metric_list = metric_list

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

    def __init__(self,config:Config,fold_id:int,metric_list:list):
        self.config = config
        self.device = torch.device(config.device)
        self.fold_id = fold_id
        self.logger = Logger(config,metric_list)

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
                loss, metric_values = compute_loss_and_metrics
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


# Train Helpers

def get_model(input_dim:int,config:Config,device):
    """ Retrieve the model based on the given config. """
    if config.model_type == 'vae':
        model = VAE(input_dim,config)
        for roland_layer in model.patient_encoder.backbone.rnn_layers:
            roland_layer.to(device)
    else:
        raise ValueError(f'Model type {config.model_type} is not implemented.')
    model.to(device)
    return model

def create_target_dict(batch,config:Config) -> dict: 
    """ Retrieve target for all lesions/patient across all time points. """
    target_dict = {'lesion':dict(),'patient':dict()}
    device = torch.device('mps')
    for t in range(config.time_steps):
        # Lesion
        volume_tensor = batch[t].x[:,1:-1] #batch[t].x[:,9:-1]
        target_dict['lesion'][t] = volume_tensor.to(device)
        # Patient
        volume_tensor = batch[t+1].x[batch[t+1].update_index,-1]
        target_dict['patient'][t] = volume_tensor.to(device)
    return target_dict

def train_one_epoch(optimizer,model,train_loader,val_loader,
                    fold_id:int,cross_val_dict:dict,
                    config:Config,temperature:float):
    """ Train function for one whole epochs. 

        :param optimizer: Pytorch optimizer to minimze the loss function.
        :param model: GNN model to make the predictions.
        :param train_loader: Graph DataLoader in batch format for training data.
        :param val_loader: Graph DataLoader in batch format for validation data.
        :param loss_fn: Pytorch loss function to be minimized.
        :param device: Device on which the computation is made.
        :param fold_id: Index of the fold kept apart as validation.
        :param cross_val_dict: Dictionnary to store evolution of values across folds.
        :param target: Choose between relapse or histological grade for the target.

    """
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
            # Retrieve volumes
            target_dict = create_target_dict(batch,config)
            # Forward Pass
            lesion_register, patient_register = model.forward(batch,temperature)
            loss, loss_dict = categorical_loss_fn(lesion_register,patient_register,target_dict,config.time_steps,config.beta)
            # Update using the gradients
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()
        # Register metrics
        update_metrics(cross_val_dict,loss_dict,split,temperature)
    # Save if save path provided
    #if config.save_path != '' and cross_val_dict[f'val_total_loss'][-1] <= np.min(cross_val_dict[f'val_total_loss']):
    torch.save(model,os.path.join(config.save_path,f'model_fold_{fold_id}_best_loss.pth'))

def train(train_loader:DataLoader,val_loader:DataLoader,input_dim:int,fold_id:int,cross_val_dict:dict,config:Config):
    """ Full end-to-end training of a model. 
    
        :param train_loader: Graph DataLoader in batch format for training data.
        :param val_loader: Graph DataLoader in batch format for validation data.
        :param input_dim: Input dimension of node features.
        :param fold_id: Index of the fold kept apart as validation.
        :param cross_val_dict: Dictionnary to store evolution of values across folds.
        :param config: Wandb config object with all training parameters.
        
    """
    # Model initialization
    device = torch.device('mps')
    model = get_model(input_dim,config,device)
    # Optimization modules
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Model training
    for epoch in range(config.epochs):
        temp = np.maximum(config.temp_0 * np.exp(-config.anneal_rate * epoch), config.temp_min)
        train_one_epoch(optimizer,model,train_loader,val_loader,
                        fold_id,cross_val_dict,config,temp)
    # Release memory
    del model

def train_one_fold(fold_id:int,config:Config,cross_val_dict:dict):
    """ Full pipeline for model training, leaving one fold for validation.
    
        :param fold_id: Index of the fold kept apart as validation.
        :param config: Wandb config object with all training parameters.
        :param cross_val_dict: Dictionnary to store evolution of values across folds.
    
    """
    # Reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    # Datasets
    train_dataset_path = os.path.join(config.data_paths[str(fold_id)],'train')
    val_dataset_path = os.path.join(config.data_paths[str(fold_id)],'val')
    training_dataset, validation_dataset = get_datasets(train_dataset_path,val_dataset_path,config)
    # Retrieve key parameters
    example_data_point = training_dataset.get(0)
    input_dim = example_data_point[0].x.shape[1]
    # DataLoaders
    training_loader = TemporalDataLoader(training_dataset,batch_size=config.batch_size,generator=generator,follow_batch=['edge_index'],shuffle=True)
    validation_loader = TemporalDataLoader(validation_dataset,batch_size=config.batch_size,generator=generator,follow_batch=['edge_index'],shuffle=True)
    # Training
    train(training_loader,validation_loader,input_dim,fold_id,cross_val_dict,config)

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

def gnn_pipeline(parameters_dict:dict,project_name:str,method:str='grid'):
    """ Hyper parameter search with cross-validation for a GNN. """
    # Process to cross validation
    std_dict = dict()
    for fold_id in range(len(parameters_dict['data_paths']['values'][0])):
        # Instantiate metric registration
        sweep_config = create_sweep_config(parameters_dict,method=method)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        # Launch training
        wandb.agent(sweep_id,partial(gnn_single_fold,fold_id=fold_id,std_dict=std_dict))
    # Log standard deviations
    sweep_config = create_sweep_config(parameters_dict,method=method)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id,partial(std_logging,std_dict=std_dict))

def gnn_single_fold(config:Config=None,fold_id:int=0,std_dict:dict=dict()):
    """ Training and metric registration for a single fold
        
        :param config: Wandb config for the hyperparameter search.
        :param fold_id: Id of the fold for this training.
        :param std_dict: Dictionnary to keep track of metric values for latter std computation.
    
    """
    with wandb.init(config=config):
        config = wandb.config
        # Perform training
        cross_val_dict = create_metric_dict(config)
        train_one_fold(fold_id,config,cross_val_dict)
        # Assign target specific variables
        metric_list = get_metric_list(config)
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
