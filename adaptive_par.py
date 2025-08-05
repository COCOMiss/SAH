import copy
import logging
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import warnings

class AdaptiveParameter():
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        update_weights_every: int = 1,
    ):
        
        self.n_tasks=n_tasks
        self.device=device
        self.update_weights_every = update_weights_every
        
        

    

    def task_specific_params(
        self,
        losses,
        model,
        specific_parameters_set
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """

        
        

        low_similarity_pairs,task_groups= find_most_divergent_parameter_module(model, losses,specific_parameters_set)
        
        for param_idx, pairs in low_similarity_pairs.items():
            logging.info("Parameter {} has low similarity pairs: {}".format(param_idx,pairs))
            
        for param_name in task_groups.keys():
            specific_parameters_set.add(param_name)
            
            
        return task_groups,specific_parameters_set



def create_task_specific_parameters(model, task_groups,task_specific_parameters,specific_parameters_set):
    """
    Create task-specific copies of a parameter for tasks with negative gradient similarity.
    Args:
        model: The model containing the parameters
        param_idx: Index of the parameter to split
        task_indices: List of task indices that need separate parameters
    Returns:
        Dictionary mapping task indices to their parameter copies
    """
    original_param=list(model.parameters())
    for param_idx in task_groups.keys():
        specific_parameters_set.add(param_idx)
        # task_params[param_idx] = {}
        
        # Create a copy of the original parameter
        for group_id in task_groups[param_idx].keys():
            if group_id==0:
                for task_idx in task_groups[param_idx][group_id]:
                    if task_idx not in task_specific_parameters:
                        task_specific_parameters[task_idx] = {}
                    task_specific_parameters[task_idx][param_idx] = original_param[param_idx]
                # task_params[param_idx][group_id] = original_param[param_idx]
            else:
                copy_param=nn.Parameter(original_param[param_idx].data.clone())
                for task_idx in task_groups[param_idx][group_id]:
                    if task_idx not in task_specific_parameters:
                        task_specific_parameters[task_idx] = {}
                    task_specific_parameters[task_idx][param_idx] = copy_param
                # task_param = torch.nn.Parameter(original_param.data.clone())
                # task_params[param_idx][group_id] = task_param
    
    return task_specific_parameters,specific_parameters_set


def get_task_groups(low_similarity_pairs,similarities):
    groups = {}
    groups[0] = set()
    groups[1] = set()
    temp_set=set()
    for par_name,task_i,task_j,similarity in low_similarity_pairs:
        if task_i not in groups[0]:
            if task_i not in groups[1]:
                groups[0].add(task_i)
                groups[1].add(task_j)
            else:
                groups[0].add(task_j)
        else:
            groups[1].add(task_j)
        temp_set.add(task_i)
        temp_set.add(task_j)
    
    for i in range(similarities.shape[0]):
        if i not in temp_set:
            max_sim=0.0
            max_group=0
            for group in groups:
                temp_sim=0.0
                for task_index in groups[group]:
                    temp_sim+=similarities[i,task_index]
                temp_sim=temp_sim/len(groups[group])
                if temp_sim>max_sim:
                    max_sim=temp_sim
                    max_group=group
            groups[max_group].add(i)
            temp_set.add(i)
         
       
    return groups

def find_most_divergent_parameter_module(model, losses,specific_parameters_set):
    """
    Find parameters with negative gradient similarity and return task-specific parameter information.
    Args:
        shared_parameters: List of parameter modules
        losses: List of losses
    Returns:
        G: Stacked gradients matrix
        low_similarity_pairs: Dictionary mapping parameter indices to list of (param_name, task_i, task_j, similarity)
        task_groups: List of task groups for each parameter
    """

    
    # Initialize dictionaries
    param_grads = {}
    # loss_grads = {i: [] for i in range(n_losses)}
    param_names = []
    low_similarity_pairs = {}
    task_groups = {}
    
    
    for index, (name, param) in enumerate(model.named_parameters()):
   
        
        param_names.append(name)
        
        param_grads[index] = []
        if index in specific_parameters_set:
            continue
        
        for loss_idx, loss in enumerate(losses):
            grad = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                grad = grad.flatten()
                grad = grad / (torch.norm(grad) + 1e-8)
                param_grads[index].append(grad)
                # loss_grads[loss_idx].append(grad)
    
    for param_idx in param_grads:
        if len(param_grads[param_idx]) <= 1:
            continue
        if param_idx in specific_parameters_set:
            continue
            
        n_grads = len(param_grads[param_idx])
        similarities = torch.zeros((n_grads, n_grads), device=losses[0].device)
        
        for i in range(n_grads):
            for j in range(i+1, n_grads):
                sim = torch.dot(param_grads[param_idx][i], param_grads[param_idx][j])
                
                similarities[i, j] = sim
                similarities[j, i] = sim
                
                if sim < -0.0001:
                    if param_idx not in low_similarity_pairs:
                        low_similarity_pairs[param_idx] = []
                    low_similarity_pairs[param_idx].append((param_names[param_idx], i, j, sim.item()))
        if param_idx in low_similarity_pairs and len(low_similarity_pairs[param_idx]) > 0:
            task_groups[param_names[param_idx]]=get_task_groups(low_similarity_pairs[param_idx],similarities)
            
    
   
    return low_similarity_pairs, task_groups
