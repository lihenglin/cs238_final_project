import torch
from torch import optim, nn
import numpy as np
import random
from typing import Callable

def get_default_train_op(model: nn.Module,
                         learning_rate: float,
                         gradient_clip: bool,
                         gradient_clip_norm: float) -> Callable[[], None]: 
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    def _train_op(): 
        if gradient_clip: 
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    return _train_op

def get_multi_modules_train_op(modules_parameters,
                               learning_rate: float,
                               gradient_clip: bool,
                               gradient_clip_norm: float) -> Callable[[], None]: 
    optimizers = [optim.Adam(parameters, lr=learning_rate) for parameters in modules_parameters]

    def _train_op(): 
        if gradient_clip: 
            for parameters in modules_parameters:
                nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

    return _train_op

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
